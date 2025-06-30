from sdo_clv_pipeline.sdo_download import *

download_data(series="720", email="mlp95@psu.edu", outdir="/Users/srugins/sdo-clv-pipeline/data", start="2014/01/01", end="2014/01/02", 
              sample=4, overwrite=False, progress=False)
