# Public API

## sdo_clv_pipeline.sdo_image
::: sdo_clv_pipeline.sdo_image
    options:
      members:
        - SDOImage
        - SunMask

## sdo_clv_pipeline.sdo_process
::: sdo_clv_pipeline.sdo_process
    options:
      members:
        - is_quality_data
        - reduce_sdo_images
        - process_data_set_parallel
        - process_data_set

## sdo_clv_pipeline.sdo_vels
::: sdo_clv_pipeline.sdo_vels
    options:
      members:
        - compute_disk_results
        - compute_region_only_results
        - compute_region_results

## sdo_clv_pipeline.sdo_plot
::: sdo_clv_pipeline.sdo_plot
    options:
      members:
        - plot_image
        - plot_mask
        - label_moats_on_sun

## sdo_clv_pipeline.sdo_download
::: sdo_clv_pipeline.sdo_download
    options:
      members:
        - download_data

## sdo_clv_pipeline.legendre
::: sdo_clv_pipeline.legendre
    options:
      members:
        - gen_leg_vec
        - gen_leg_x_vec

## sdo_clv_pipeline.reproject
::: sdo_clv_pipeline.reproject
    options:
      members:
        - compute_pixel_mapping
        - bilinear_reproject