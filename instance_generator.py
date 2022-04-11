import pandas as pd
from shapely.geometry import Point
import numpy as np
import geopandas as gpd
import os
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import seaborn as sns
import scipy.stats as st


np.random.seed(0)


def read_and_preprocess_cities(plot=False):
    """Reading and preprocessing the cities data frame"""
    cities = gpd.read_file(os.path.join(data_dir, "cities.geojson"))
    cities = cities[cities.intersects(turkey_polygon.buffer(0.005))]
    columns_to_keep = ["name", "population", "geometry"]
    cities.drop(columns=[c for c in cities.columns if c not in columns_to_keep], inplace=True)
    cities = cities[~cities["population"].isna()]
    cities["population"] = cities["population"].astype(int)
    cities = cities.sort_values("population", ascending=False)
    cities.reset_index(inplace=True, drop=True)
    cities.to_file(os.path.join(data_dir, "turkey_cities.geojson"), driver="GeoJSON")
    if plot:
        ax = turkey.plot()
        cities.plot(ax=ax, color="red")
        plt.show()
    return cities


# Static Data loading
data_dir = "data/"
turkey = gpd.read_file(os.path.join(data_dir, "turkey.geojson"))
turkey_polygon = turkey.iloc[0]["geometry"]

turkey_cities_path = os.path.join(data_dir, "turkey_cities.geojson")
if os.path.exists(turkey_cities_path):
    cities = gpd.read_file(os.path.join(data_dir, "turkey_cities.geojson"))
else:
    cities = read_and_preprocess_cities()

coords = [[p.x, p.y] for p in cities["geometry"]]
cities_spatial_tree = cKDTree(coords)
kde = st.gaussian_kde(np.transpose(coords), weights=list(cities["population"]))


def rand_float(a: float, b: float) -> float:
    """Get a random float between two bounds with numpy"""
    return np.random.rand() * (b - a) + a


def uniform_2d():
    """Return a random uniformly distributed point within Turkey polygon bounding box"""
    xmin, ymin, xmax, ymax = turkey_polygon.bounds
    x = rand_float(xmin, xmax)
    y = rand_float(ymin, ymax)
    return x, y


def kde_2d():
    """Return a random point based on a given KDE distribution"""
    res = kde.resample(1)
    return res[0][0], res[1][0]


def get_random_point_in_polygon(polygon, sample_method) -> Point:
    """Return a random point within a given polygon with a given random distribution"""
    while True:
        x, y = sample_method()
        p = Point(x, y)
        if p.within(polygon):
            return p


def distance_between_two_points(a: Point, b: Point) -> float:
    """Simple euclidian distance"""
    return ((a.x - b.x)**2 + (a.y - b.y)**2)**0.5


def gen_demands(N, dem_min, dem_max, method="proportional_to_cities_pop"):
    """Generating the demands points"""
    demands = gpd.GeoDataFrame(columns=["Demand", "geometry"])
    min_pop, max_pop = cities["population"].min(), cities["population"].max()
    for i in range(N):
        p = get_random_point_in_polygon(turkey_polygon, kde_2d)
        if method == "random":
            d = np.random.randint(dem_min, dem_max)
        elif method == "proportional_to_cities_pop":
            _, closest_cities = cities_spatial_tree.query([p.x, p.y], k=5)
            mean_pop = cities.iloc[closest_cities]["population"].mean()
            d = int(dem_min + (dem_max - dem_min) * (mean_pop - min_pop) / max_pop)
        else:
            raise ValueError("Unknown method")
        demands.loc[len(demands)] = [d, p]
    demands["Demand"] = demands["Demand"].astype(int)
    return demands


def vary_demands(retailers, nb_scenarios, variance=20):
    """Make some alternative demnad scenario based on a normal distribution with a given variance"""
    retailers = retailers.copy()
    for s in range(nb_scenarios):
        retailers["Demand%d" % s] = np.random.normal(retailers["Demand"], variance)
    retailers.drop(columns=["Demand"], inplace=True)
    return retailers


def plot_scenarios(retailers, nb_scenarios):
    """Plot difference scenarios of demands"""
    retailers_records = pd.DataFrame(columns=["retailer", "Demand", "scenario"])
    retailers_records["retailer"] = list(retailers.index) * nb_scenarios
    retailers_records["Demand"] = [retailers.at[r, "Demand%d" % s] for r in retailers.index for s in range(nb_scenarios)]
    retailers_records["scenario"] = [s for r in retailers.index for s in range(nb_scenarios)]
    sns.boxplot(x="retailer", y="Demand", data=retailers_records)


def gen_warehouses(M, capa_min, capa_max, capex_nominal, method="random"):
    """Generate the warehouses data"""
    warehouses = gpd.GeoDataFrame(columns=["Capacity", "CAPEX", "geometry"])
    min_pop, max_pop = cities["population"].min(), cities["population"].max()
    for i in range(M):
        p = get_random_point_in_polygon(turkey_polygon, kde_2d)
        if method == "random":
            capa = np.random.randint(capa_min, capa_max)
        elif method == "proportional_to_cities_pop":
            _, closest_cities = cities_spatial_tree.query([p.x, p.y], k=5)
            mean_pop = cities.iloc[closest_cities]["population"].mean()
            capa = int(capa_min + (capa_max - capa_min) * (mean_pop - min_pop) / max_pop)
        else:
            raise ValueError("Unknown method")
        capex = capa * rand_float(0.75, 1.25) * capex_nominal
        warehouses.loc[len(warehouses)] = [capa, capex, p]
    warehouses["Capacity"] = warehouses["Capacity"].astype(int)
    warehouses["CAPEX"] = warehouses["CAPEX"].astype(int)
    return warehouses


def plot_capacities(warehouses):
    """Simple plot of the capacities of warehouses dataset"""
    ax = turkey.plot(alpha=0.5)
    cities.plot(ax=ax, color="red")
    warehouses.plot(ax=ax, column="Capacity", legend=True)
    plt.show()


def gen_costs(retailers, warehouses, opex_nominal=100):
    """Generate the opex costs"""
    distances = [[distance_between_two_points(a, b) for a in warehouses["geometry"]] for b in retailers["geometry"]]
    max_dist = max(max(ds) for ds in distances)
    costs = pd.DataFrame(columns=warehouses.index)
    for r, rn in enumerate(retailers.index):
        costs.loc[rn] = [rand_float(0.9, 1.1) * distances[r][w] / max_dist * opex_nominal for w in range(len(warehouses.index))]
    return costs


def gen_instance(name_prefix: str, **kwargs):
    """Main method to generate a FLP instance"""
    N = kwargs.get("N", 500)
    M = kwargs.get("M", 50)
    dem_min = kwargs.get("dem_min", 100)
    dem_max = kwargs.get("dem_max", 1000)
    capa_min = kwargs.get("capa_min", 15000)
    capa_max = kwargs.get("capa_max", 30000)
    capex_nominal = kwargs.get("capex_nominal", 1000)
    opex_nominal = kwargs.get("opex_nominal", 5000)
    nb_scenarios = kwargs.get("nb_scenarios", None)
    if nb_scenarios is None:
        out_dir = "data/%s_%d_%d/" % (name_prefix, N, M)
    else:
        out_dir = "data/%s_%d_%d_%d/" % (name_prefix, N, M, nb_scenarios)
    os.makedirs(out_dir, exist_ok=True)
    retailers = gen_demands(N=N, dem_min=dem_min, dem_max=dem_max)
    retailers.to_file(os.path.join(out_dir, "retailers.geojson"), driver="GeoJSON")
    if nb_scenarios is not None:
        retailers2 = vary_demands(retailers, nb_scenarios)
        retailers2.to_file(os.path.join(out_dir, "retailers.geojson"), driver="GeoJSON")

    warehouses = gen_warehouses(M=M, capa_min=capa_min, capa_max=capa_max, capex_nominal=capex_nominal)
    warehouses.to_file(os.path.join(out_dir, "warehouses.geojson"), driver="GeoJSON")
    opex = gen_costs(retailers, warehouses, opex_nominal=opex_nominal)
    opex.to_csv(os.path.join(out_dir, "opex_costs.csv"), sep=";")


if __name__ == "__main__":
    gen_instance("12")
