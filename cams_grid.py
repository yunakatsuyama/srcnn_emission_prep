import numpy as np

class CAMSExtractor:
    """
    Handle CAMS dataset and extract subgrid
    """

    def __init__(self, camsds):
        self.ds = camsds.squeeze()

        self.lon = self.ds['lon'].values
        self.lat = self.ds['lat'].values
        self.val = 1e9 * 1e-2 * self.ds['sum'].values

    def extract_window(self, lon_center, lat_center, pix=5):
        """
        Extract NxN window around given location
        """

        ix0 = np.argmin(np.abs(self.lon - lon_center))
        iy0 = np.argmin(np.abs(self.lat - lat_center))

        half = pix // 2

        ix = np.arange(ix0 - half, ix0 + half + 1)
        iy = np.arange(iy0 - half, iy0 + half + 1)

        ix = ix[(ix >= 0) & (ix < len(self.lon))]
        iy = iy[(iy >= 0) & (iy < len(self.lat))]

        cams_lon_array = self.lon[ix]
        cams_lat_array = self.lat[iy]
        cams_emission = self.val[np.ix_(iy, ix)]

        return cams_lon_array, cams_lat_array, cams_emission
    