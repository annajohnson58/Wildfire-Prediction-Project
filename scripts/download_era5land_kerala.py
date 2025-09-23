import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-land-monthly-means',
    {
        'format': 'netcdf',
        'variable': [
            '2m_temperature',
            'total_precipitation',
            '10m_u_component_of_wind',
            '10m_v_component_of_wind'
        ],
        'year': '2024',
        'month': [f'{i:02d}' for i in range(1, 13)],
        'area': [11.0, 74.5, 8.0, 77.5],  # Kerala bounding box: N, W, S, E
        'time': '00:00',
    },
    'data/era5land_kerala_2024.nc'
)
