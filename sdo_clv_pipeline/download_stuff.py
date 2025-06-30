from sdo_clv_pipeline.sdo_download import *

data_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "data"))
download_data(series="720", email="mlp95@psu.edu", outdir=data_dir, start="2014/01/01", end="2014/01/02", 
              sample=4, overwrite=False, progress=False)
