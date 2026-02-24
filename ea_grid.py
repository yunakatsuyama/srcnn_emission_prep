# grids.py
import numpy as np
import xesmf as xe



class EAGridRegridder:
    """
    Handle:
        - EAGrid → fine grid (conservative)
        - CAMS → fine grid (bilinear)
        - CAMS → fine grid (conservative)
    """

    def __init__(self, ds_eagrid, cams_lon_array, cams_lat_array, cams_emission, k):
        """
        Parameters
        ----------
        ds_eagrid : xarray.Dataset
            Emission_grid_all.nc dataset
        cams_lon, cams_lat : 0.1 degree, defined at the center of grid boxes.
        cams_emission : 2D array (5 x 5)
        k : downscale factor

        """
        self.ds = ds_eagrid
        self.cams_lon_array = cams_lon_array
        self.cams_lat_array = cams_lat_array
        self.cams_emission = cams_emission
        self.k = k

        # placeholders
        self.eagrid_lon = None
        self.eagrid_lat = None

        self.eagrid_conserv = None
        self.cams_bilinear = None
        self.cams_conserv = None

    # --------------------------------------------------
    # 1. Build fine grid
    # --------------------------------------------------
    def build_fine_grid(self):
        """
        Based on cams grid, the fined target grid eagrid_lon, eagrid_lat (50x50), defined at the center of grid boxes.
        """

        dlon = np.abs(self.cams_lon_array[1] - self.cams_lon_array[0])
        dlat = np.abs(self.cams_lat_array[1] - self.cams_lat_array[0])

        lon_min = self.cams_lon_array[0] - dlon / 2
        lon_max = self.cams_lon_array[-1] + dlon / 2
        lat_min = self.cams_lat_array[0] - dlat / 2
        lat_max = self.cams_lat_array[-1] + dlat / 2

        fine_dlon = dlon / self.k
        fine_dlat = dlat / self.k

        self.eagrid_lon = np.linspace(
            lon_min + fine_dlon/2,
            lon_max - fine_dlon/2,
            len(self.cams_lon_array) * self.k
        )

        self.eagrid_lat = np.linspace(
            lat_min + fine_dlat/2,
            lat_max - fine_dlat/2,
            len(self.cams_lat_array) * self.k
        )

    # --------------------------------------------------
    # 2. Extract EAGrid emission
    # --------------------------------------------------
    def extract_eagrid(self, sector="All_sources", source="NOx"):

        sec_idx = int(self.ds.sector.to_index().get_loc(sector))
        src_idx = int(self.ds.source.to_index().get_loc(source))

        lat_e = self.ds["lat"].values
        lon_e = self.ds["lon"].values
        emission = self.ds["Emission"][sec_idx, src_idx, :].values

        # mask
        margin = 0.05
        self.lat_min, self.lat_max = self.eagrid_lat[0], self.eagrid_lat[-1]
        self.lon_min, self.lon_max = self.eagrid_lon[0], self.eagrid_lon[-1]

        mask = (lat_e >= self.lat_min - margin) & (lat_e <= self.lat_max + margin) & \
           (lon_e >= self.lon_min - margin) & (lon_e <= self.lon_max + margin)
    # extract # lat_in, lon_in are left bottom
        self.lat_in = lat_e[mask]
        self.lon_in = lon_e[mask]
        self.emission_in = emission[mask]

    # --------------------------------------------------
    # 3. Regridding
    # --------------------------------------------------
    def regrid_all(self):
        # eagrid_lon,lon is target 50x50 grid defined at the center
        # lat_t_edges, lon_t_edges are the target 51x51 defined at the edges. 
        # target grid
        lat_t_edges = self.centers_to_edges(self.eagrid_lat)
        lon_t_edges = self.centers_to_edges(self.eagrid_lon)

        grid_out = {
            "lon": self.eagrid_lon,
            "lat": self.eagrid_lat,
            "lon_b": lon_t_edges,
            "lat_b": lat_t_edges
        }

        # =============================
        # CAMS bilinear
        # =============================
        cams_lat_edges = self.centers_to_edges(self.cams_lat_array)
        cams_lon_edges = self.centers_to_edges(self.cams_lon_array)

        grid_in_cams = {
            "lon": self.cams_lon_array,
            "lat": self.cams_lat_array,
            "lon_b": cams_lon_edges,
            "lat_b": cams_lat_edges
        }

        regrid_bilin = xe.Regridder(
            grid_in_cams,
            grid_out,
            "bilinear",
            extrap_method="nearest_s2d"
        )

        self.cams_bilinear = regrid_bilin(self.cams_emission)  # the oridinal code, it was fine_cams_emission_interp

        # =============================
        # CAMS conservative
        # =============================
        regrid_cons = xe.Regridder(
            grid_in_cams,
            grid_out,
            "conservative"
        )

        self.cams_conserv = regrid_cons(self.cams_emission)

        # =============================
        # EAGrid conservative
        # =============================
        # emission_grid must be prepared same as before
        self.prepare_eagrid_grid()
        self.build_eagrid_geometry(self.lat_min, self.lat_max, self.lon_min, self.lon_max)
        
        grid_in_ea = {
                "lon": self.lon_e_center,
                "lat": self.lat_e_center,
                "lon_b": self.lon_e_edges_xe,
                "lat_b": self.lat_e_edges_xe
            }

        # I need lon_e_center and lon_e_edges_xe 
        regrid_ea = xe.Regridder(
            grid_in_ea,
            grid_out,
            "conservative"
        )

        self.eagrid_conserv = regrid_ea(self.emission_grid)

    # --------------------------------------------------
    # helper functions
    # --------------------------------------------------
    def prepare_eagrid_grid(self):
        """
        Convert point-based emission into 2D grid
        """

        Lon_in, Lat_in = np.meshgrid(
            self.lon_e_edges,
            self.lat_e_edges
        )

        emission_grid = np.full(Lon_in.shape, np.nan)

        for lon, lat, val in zip(
            self.lon_in,
            self.lat_in,
            self.emission_in
        ):
            i = np.argmin(np.abs(self.lat_e_edges - lat))
            j = np.argmin(np.abs(self.lon_e_edges - lon))
            emission_grid[i, j] = val

        emission_grid = np.nan_to_num(emission_grid, nan=0.0)

        self.emission_grid = emission_grid


    def build_eagrid_geometry(self, lat_min, lat_max, lon_min, lon_max):
        """
        Build EAGrid edges and centers
        """

        lon_e_edges, lat_e_edges = self.eagrid_latlon_array(
            [lat_min, lat_max],
            [lon_min, lon_max]
        )

        # hard-coded resolution (should later be improved)
        dlon_e = 1 / 160
        dlat_e = 1 / 240

        self.lon_e_edges = lon_e_edges
        self.lat_e_edges = lat_e_edges

        self.lon_e_center = lon_e_edges + dlon_e
        self.lat_e_center = lat_e_edges + dlat_e

        # add upper bound for xESMF
        self.lon_e_edges_xe = np.append(
            lon_e_edges,
            lon_e_edges[-1] + 2 * dlon_e
        )

        self.lat_e_edges_xe = np.append(
            lat_e_edges,
            lat_e_edges[-1] + 2 * dlat_e
        )
    
    def centers_to_edges(arr):
        d = np.diff(arr)
        edges = np.empty(len(arr) + 1)
        edges[1:-1] = arr[:-1] + d / 2
        edges[0] = arr[0] - d[0] / 2
        edges[-1] = arr[-1] + d[-1] / 2
        return edges
    
    def eagrid_latlon_array(lat_range,lon_range):  # lon_range = [lon_min, lon_max], lat_range = [lat_min, lat_max]
     
        lat_range = np.array(lat_range)
        lon_range  = np.array(lon_range)
        
        lat_range_grid = np.ceil(lat_range * 120) * 1/120
        lon_range_grid = np.ceil(lon_range * 80) * 1/80

        #print(lat_range_grid)
        #print(lon_range_grid)
        
        lat_array = np.arange(lat_range_grid[0],lat_range_grid[1] + 1/120,  1/120 )
        lon_array  = np.arange(lon_range_grid[0],lon_range_grid[1] + 1/120 ,  1/80 )
        
        lat_array = np.round(lat_array, 5)
        lon_array = np.round(lon_array ,5)

        return lon_array, lat_array