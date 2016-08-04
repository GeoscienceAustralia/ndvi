from __future__ import absolute_import, print_function

import logging
import os
from copy import deepcopy
from datetime import datetime
import functools

import click
from pandas import to_datetime
from pathlib import Path
import xarray

from datacube.api.grid_workflow import GridWorkflow
from datacube.model import DatasetType, GeoPolygon, Range
from datacube.model.utils import make_dataset, xr_apply, datasets_to_doc
from datacube.storage.storage import write_dataset_to_netcdf
from datacube.storage.masking import mask_valid_data
from datacube.ui import click as ui
from datacube.ui.task_app import task_app, task_app_options, get_full_lineage
from datacube.utils import intersect_points, union_points


_LOG = logging.getLogger('agdc-ndvi')


def make_ndvi_config(index, config, **query):
    dry_run = query.get('dry_run', False)
    config['overwrite'] = query.get('overwrite', False)

    source_type = index.products.get_by_name(config['source_type'])
    if not source_type:
        _LOG.error("Source DatasetType %s does not exist", config['source_type'])
        return 1

    output_type_definition = deepcopy(source_type.definition)
    output_type_definition['name'] = config['output_type']
    output_type_definition['managed'] = True
    output_type_definition['description'] = config['description']
    output_type_definition['storage'] = config['storage']
    output_type_definition['metadata']['format'] = {'name': 'NetCDF'}
    output_type_definition['metadata']['product_type'] = config.get('product_type', 'ndvi')

    var_def_keys = {'name', 'dtype', 'nodata', 'units', 'aliases', 'spectral_definition', 'flags_definition'}

    output_type_definition['measurements'] = [{k: v for k, v in measurement.items() if k in var_def_keys}
                                              for measurement in config['measurements']]

    chunking = config['storage']['chunking']
    chunking = [chunking[dim] for dim in config['storage']['dimension_order']]

    var_param_keys = {'zlib', 'complevel', 'shuffle', 'fletcher32', 'contiguous', 'attrs'}
    variable_params = {}
    for mapping in config['measurements']:
        varname = mapping['name']
        variable_params[varname] = {k: v for k, v in mapping.items() if k in var_param_keys}
        variable_params[varname]['chunksizes'] = chunking

    config['variable_params'] = variable_params

    output_type = DatasetType(source_type.metadata_type, output_type_definition)

    if not dry_run:
        _LOG.info('Created DatasetType %s', output_type.name)
        output_type = index.products.add(output_type)

    if not os.access(config['location'], os.W_OK):
        _LOG.warn('Current user appears not have write access output location: %s', config['location'])

    config['nbar_dataset_type'] = source_type
    config['ndvi_dataset_type'] = output_type

    return config


def get_filename(config, tile_index, sources):
    file_path_template = str(Path(config['location'], config['file_path_template']))
    return file_path_template.format(tile_index=tile_index,
                                     start_time=to_datetime(sources.time.values[0]).strftime('%Y%m%d%H%M%S%f'),
                                     end_time=to_datetime(sources.time.values[-1]).strftime('%Y%m%d%H%M%S%f'))


def make_ndvi_tasks(index, config, **kwargs):
    input_type = config['nbar_dataset_type']
    output_type = config['ndvi_dataset_type']

    workflow = GridWorkflow(index, output_type.grid_spec)

    # TODO: Filter query to valid options

    query = {}
    if 'year' in kwargs:
        year = int(kwargs['year'])
        query['time'] = Range(datetime(year=year, month=1, day=1), datetime(year=year+1, month=1, day=1))

    tiles_in = workflow.list_tiles(product=input_type.name, **query)
    tiles_out = workflow.list_tiles(product=output_type.name, **query)

    # TODO: Move get_full_lineage & update_sources to GridWorkflow/Datacube/model?
    def update_sources(sources):
        return tuple(get_full_lineage(index, dataset.id) for dataset in sources)

    def update_tile(tile):
        for i in range(tile['sources'].size):
            tile['sources'].values[i] = update_sources(tile['sources'].values[i])
        return tile

    def make_task(tile, **kwargs):
        nbar = update_tile(tile.copy())
        task = {
            'nbar': nbar
        }
        task.update(kwargs)
        return task

    tasks = [make_task(tile, tile_index=key, filename=get_filename(config, tile_index=key, sources=tile['sources']))
             for key, tile in tiles_in.items() if key not in tiles_out]

    _LOG.info('%s tasks discovered', len(tasks))
    return tasks


def get_app_metadata(config):
    doc = {
        'lineage': {
            'algorithm': {
                'name': 'datacube-ndvi',
                'version': config.get('version', 'unknown'),
                'repo_url': 'https://github.com/GeoscienceAustralia/ndvi.git',
                'parameters': {'configuration_file': config.get('app_config_file', 'unknown')}
            },
        }
    }
    return doc


def do_ndvi_task(config, task):
    output_type = config['ndvi_dataset_type']
    nodata_value = output_type.definition['measurements']['ndvi'].nodata
    output_dtype = output_type.definition['measurements']['ndvi'].dtype
    output_units = output_type.definition['measurements']['ndvi'].units
    output_scale = 10000
    nodata_value = output_dtype.type(nodata_value)

    measurements = ['red', 'nir']

    nbar = GridWorkflow.load(task['nbar'], measurements)

    nbar_masked = mask_valid_data(nbar)
    ndvi_array = (nbar_masked.nir - nbar_masked.red) / (nbar_masked.nir + nbar_masked.red)
    ndvi_out = (ndvi_array * output_scale).fillna(nodata_value).astype(output_dtype)
    ndvi_out.attrs = {
        'crs': nbar.attrs['crs'],
        'units': output_units,
        'nodata': nodata_value,
    }

    ndvi = xarray.Dataset(ndvi_out, dims=nbar.red.dims, coords=nbar.coords, attrs=nbar.attrs)

    global_attributes = config['global_attributes']
    variable_params = config['variable_params']
    file_path = Path(task['filename'])
    output_type = config['ndvi_dataset_type']

    def _make_dataset(labels, sources):
        assert len(sources)
        geobox = task['nbar']['geobox']
        source_data = functools.reduce(union_points, (dataset.extent.to_crs(geobox.crs).points for dataset in sources))
        valid_data = intersect_points(geobox.extent.points, source_data)
        dataset = make_dataset(dataset_type=output_type,
                               sources=sources,
                               extent=geobox.extent,
                               center_time=labels['time'],
                               uri=file_path.absolute().as_uri(),
                               app_info=get_app_metadata(config),
                               valid_data=GeoPolygon(valid_data, geobox.crs))
        return dataset
    sources = task['nbar']['sources']
    datasets = xr_apply(sources, _make_dataset, dtype='O')
    ndvi['dataset'] = datasets_to_doc(datasets)

    if config.get('overwrite', False) and file_path.exists():
        file_path.unlink()

    write_dataset_to_netcdf(ndvi, global_attributes, variable_params, Path(file_path))

    return datasets


app_name = 'ndvi'


@click.command(name=app_name)
@ui.pass_index(app_name=app_name)
@click.option('--dry-run', is_flag=True, default=False, help='Check if everything is ok')
@click.option('--overwrite', is_flag=True, default=False, help='Overwrite existing (un-indexed) files')
@click.option('--year', type=click.IntRange(1960, 2060), help='Limit the process to a particular year')
@task_app_options
@task_app(make_config=make_ndvi_config, make_tasks=make_ndvi_tasks)
def ndvi_app(index, config, tasks, executor, dry_run, *ars, **kwargs):
    click.echo('Starting NDVI processing...')

    results = []
    for task in tasks:
        click.echo('Running task: {}'.format(task))
        if not dry_run:
            results.append(executor.submit(do_ndvi_task, config=config, task=task))

    successful = failed = 0
    for result in executor.as_completed(results):
        try:
            datasets = executor.result(result)
            for dataset in datasets.values:
                index.datasets.add(dataset)
            successful += 1
        except Exception as err:  # pylint: disable=broad-except
            _LOG.exception('Task failed: %s', err)
            failed += 1
            continue

    click.echo('%d successful, %d failed' % (successful, failed))
