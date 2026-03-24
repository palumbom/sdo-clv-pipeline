Preprocessing
=============

Image data types, downloading, processing, and reprojection.

.. dropdown:: sdo_clv_pipeline.sdo_image
   :open:

   .. autoclass:: sdo_clv_pipeline.sdo_image.SDOImage
      :members:
      :show-inheritance:

   .. autoclass:: sdo_clv_pipeline.sdo_image.SunMask
      :members:
      :show-inheritance:

.. dropdown:: sdo_clv_pipeline.sdo_download

   .. autofunction:: sdo_clv_pipeline.sdo_download.download_data

.. dropdown:: sdo_clv_pipeline.sdo_process

   .. autofunction:: sdo_clv_pipeline.sdo_process.is_quality_data

   .. autofunction:: sdo_clv_pipeline.sdo_process.reduce_sdo_images

   .. autofunction:: sdo_clv_pipeline.sdo_process.process_data_set_parallel

   .. autofunction:: sdo_clv_pipeline.sdo_process.process_data_set

.. dropdown:: sdo_clv_pipeline.reproject

   .. autofunction:: sdo_clv_pipeline.reproject.compute_pixel_mapping

   .. autofunction:: sdo_clv_pipeline.reproject.bilinear_reproject
