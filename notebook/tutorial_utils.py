# This file defines util methods for the MIP tutorial
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import shapely.geometry
import numpy as np

turkey_gdf = None
data_dir = "../data"
COMPARE_TOL = 1e-4


def read_instance(name: str):
    """Read an FLP instance from its name"""
    # Read retailers
    d = os.path.join(data_dir, name)
    retailers = gpd.read_file(os.path.join(d, "retailers.geojson"))
    # Read warehouses
    warehouses = gpd.read_file(os.path.join(d, "warehouses.geojson"))
    # Read opex costs
    opex = pd.read_csv(os.path.join(d, "opex_costs.csv"), sep=";", index_col=0)
    opex.columns = list(map(int, opex.columns))
    return retailers, warehouses, opex


def get_turkey_gdf():
    """Return the turkey geodataframe and load it if not done before"""
    global turkey_gdf
    if turkey_gdf is not None:
        return turkey_gdf
    turkey_gdf = gpd.read_file(os.path.join(data_dir, "turkey.geojson"))
    return turkey_gdf


def plot_in_turkey(gdf, **kwargs):
    """Plot the given geodataframe with Turkey as back image"""
    f = plt.figure(figsize=(16, 9))
    ax = f.add_subplot()
    get_turkey_gdf().plot(ax=ax, alpha=0.5)
    # cities.plot(ax=ax, color="red")
    gdf.plot(ax=ax, **kwargs)
    plt.show()


def plot_opex_costs(retailers, warehouses, opex, to_plot, **kwargs):
    """Plot the opex costs of some warehouses"""
    sub_warehouses = warehouses[warehouses.index.isin(to_plot)]
    opex_gdf = gpd.GeoDataFrame(columns=["warehouse", "retailer", "opex", "geometry"])
    opex_gdf["warehouse"] = [w for r in retailers.index for w in sub_warehouses.index]
    opex_gdf["retailer"] = [r for r in retailers.index for w in sub_warehouses.index]
    for i in opex_gdf.index:
        w = opex_gdf.at[i, "warehouse"]
        r = opex_gdf.at[i, "retailer"]
        wp = sub_warehouses.at[w, "geometry"]
        rp = retailers.at[r, "geometry"]
        line = shapely.geometry.LineString([[wp.x, wp.y], [rp.x, rp.y]])
        opex_gdf.at[i, "opex"] = opex.at[r, w]
        opex_gdf.at[i, "geometry"] = line
    opex_gdf["opex"] = opex_gdf["opex"].astype(float)
    plot_in_turkey(opex_gdf, column="opex", legend=True, cmap="plasma",
                   legend_kwds={'label': "OPEX costs for warehouses %s" % str(to_plot),
                                'orientation': "horizontal"}, lw=0.5)


def check_solution(warehouses, retailers, opex, ysol, print_sol_cost) -> bool:
    """Check a given solution for the FLP problem"""
    if len(ysol) != len(retailers) or any(len(ysol[rix]) != len(warehouses) for rix in range(len(retailers))):
        print("Error: dimensions mismatch in input solution and number of retailers/warehouses")
        return False
    # Check that all demand is satisfied
    for rix, r in enumerate(retailers.index):
        supplied_demand = sum(ysol[rix][wix] for wix, w in enumerate(warehouses.index))
        if (retailers.at[r, "Demand"] - supplied_demand) > COMPARE_TOL:
            print("Error: Retailer %s is not fully delivered: %f < %f" % (r,  supplied_demand,
                                                                          retailers.at[r, "Demand"]))
    # Check that capacities are respected
    for wix, w in enumerate(warehouses.index):
        capa = warehouses.at[w, "Capacity"]
        supplied_demand = sum(ysol[rix][wix] for rix, w in enumerate(retailers.index))
        if supplied_demand - capa > COMPARE_TOL:
            print("Error: Warehouse %s is exceeding its capacity: %f > %f" % (w, supplied_demand, capa))
    capex_cost = sum(warehouses.at[w, "CAPEX"] for w in warehouses.index
                         if any(ysol[r][w] > COMPARE_TOL for r in retailers.index))
    opex_cost = sum(opex.at[r, w] * ysol[r][w] for w in warehouses.index for r in retailers.index)
    if print_sol_cost:
        print("Total cost=%.2f€    CAPEX=%.2f€    OPEX=%.2f€" % (capex_cost + opex_cost, capex_cost, opex_cost))
    return True


def plot_solution(warehouses, retailers, opex, ysol, print_sol_cost=True):
    """Plot a given solution"""
    if not check_solution(warehouses, retailers, opex, ysol, print_sol_cost):
        return
    sol_df = gpd.GeoDataFrame(columns=["warehouse", "retailer", "flow", "geometry"])
    non_zero_pairs = [(rix, wix, r, w) for rix, r in enumerate(retailers.index)
                      for wix, w in enumerate(warehouses.index) if ysol[rix][wix] > COMPARE_TOL]
    sol_df["warehouse"] = [w for _, _, r, w in non_zero_pairs]
    sol_df["retailer"] = [r for _, _, r, w in non_zero_pairs]
    sol_df["flow"] = [ysol[rix][wix] for rix, wix, _, _ in non_zero_pairs]
    sol_df["geometry"] = [shapely.geometry.LineString([warehouses.at[w, "geometry"],
                                                       retailers.at[r, "geometry"]])
                          for _, _, r, w in non_zero_pairs]
    sol_df["flow"] = sol_df["flow"].astype(float)
    plot_in_turkey(sol_df, column="warehouse", cmap="plasma",
                   # legend=True
                   # legend_kwds={'label': "Solution distribution network",
                   #              'orientation': "horizontal"},
                   lw=0.5)


def plot_warehouse_usage(retailers, warehouses, opex, ysol):
    """Plot the selected warehouse usage from a solution in a barplots"""
    used_warehouses = [w for w in warehouses.index if any(yval > 0.0 for r in retailers.index for yval in ysol[r][w])]
    sub_warehouses = warehouses.loc[used_warehouses]
    warehouse_use = pd.DataFrame(columns=["warehouse", "scenario", "usage"])
    non_zero_flows = [(w, s) for w in sub_warehouses.index for s in range(len(ysol[0][w]))]
    warehouse_use["usage"] = [sum(ysol[r][w][s] for r in retailers.index) for w, s in non_zero_flows]
    warehouse_use["warehouse"] = [w for w, s in non_zero_flows]
    warehouse_use["scenario"] = [s for w, s in non_zero_flows]

    import seaborn as sns
    plt.figure(figsize=(16, 9))
    sns.barplot(x="warehouse", y="usage", hue="scenario", data=warehouse_use)
    plt.scatter(range(len(sub_warehouses.index)), sub_warehouses["Capacity"], color="red", marker="^", zorder=10)


def read_quadratic_instance(name):
    """Read a quadratic FLP instance from its name"""
    d = os.path.join(data_dir, name)
    retailers, warehouses, opex = read_instance(name)
    opex_quad = pd.read_csv(os.path.join(d, "opex_quad_costs.csv"), sep=";", index_col=0)
    opex_quad.columns = list(map(int, opex_quad.columns))
    return retailers, warehouses, opex, opex_quad


def plot_quadratic_opex_costs(retailers, warehouses, opex, opex_quad, w_to_plot, r_to_plot, **kwargs):
    """Plot quadratic costs of a quadratic FLP instance"""
    f = plt.figure(figsize=(16, 9))
    ax = f.add_subplot()
    for w in w_to_plot:
        for r in r_to_plot:
            demand_r = retailers.at[r, "Demand"]
            d = np.linspace(0, demand_r, 100)
            c = d * opex.at[r, w] + d**2 * opex_quad.at[r, w]
            plt.plot(d, c, ax=ax)


def plot_solution_quad(warehouses, retailers, opex, opex_quad, ysol):
    """Plot solution for quadratic instance"""
    plot_solution(warehouses, retailers, opex, ysol, print_sol_cost=False)
    capex_cost = sum(warehouses.at[w, "CAPEX"] for w in warehouses.index
                         if any(ysol[r][w] > COMPARE_TOL for r in retailers.index))
    opex_cost = sum(opex.at[r, w] * ysol[r][w] +
                    opex_quad.at[r, w] * ysol[r][w]**2 for w in warehouses.index for r in retailers.index)
    print("Total cost=%.2f€    CAPEX=%.2f€    OPEX=%.2f€" % (capex_cost + opex_cost, capex_cost, opex_cost))
