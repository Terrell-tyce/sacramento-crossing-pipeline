"""Generate crossing coordinates for Sacramento using OSM"""
import sys
print(sys.executable)
print(sys.prefix)

import osmnx as ox
import pandas as pd
import pathlib
from pathlib import Path
import logging
import geopandas as gpd
from shapely.geometry import Point

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_sacramento_crossing_coords():
    """
    Generate crossing coordinates in Sacramento based on pedestrian crossing features.
    Following the methodology from the paper:
    - Extract all pedestrian crossing edges from OpenStreetMap
    - Use crossing nodes and edges as intersection points to analyze
    """
    
    logger.info("Fetching Sacramento boundary from OpenStreetMap...")
    
    try:
        # Get Sacramento city boundary
        logger.info("Using ox.geocode_to_gdf to fetch Sacramento boundary...")
        sacramento = ox.geocode_to_gdf("Sacramento, California, USA")
        
        # Debug: Print what we got
        geom = sacramento.geometry.iloc[0]
        logger.info(f"Geometry type: {geom.geom_type}")
        logger.info(f"Geometry bounds: {geom.bounds}")
        logger.info(f"Area (sq degrees): {geom.area:.6f}")
        
        # Convert to projected CRS to get area in square meters
        sacramento_proj = sacramento.to_crs('EPSG:3857')
        area_sq_km = sacramento_proj.geometry.iloc[0].area / 1e6
        logger.info(f"Sacramento area: {area_sq_km:.2f} sq km")
        
        # Extract ALL pedestrian crossing ways/edges from OSM
        logger.info("Extracting pedestrian crossing edges from OpenStreetMap...")
        # This catches: highway=crossing OR highway=pedestrian OR footway=crossing
        crossing_tags = {'highway': ['crossing', 'pedestrian'],}
        
        try:
            crossings = ox.features_from_polygon(
                sacramento.geometry.iloc[0],
                tags=crossing_tags
            )
            logger.info(f"Found {len(crossings)} crossing features")
        except Exception as e:
            logger.warning(f"Could not fetch crossing features: {e}")
            crossings = None
        
        # Also get all street intersections to use as fallback
        logger.info("Downloading Sacramento street network from OSM...")
        G = ox.graph_from_polygon(
            sacramento.geometry.iloc[0],
            network_type='all',
            simplify=True,
            retain_all=False,
            truncate_by_edge=True
        )
        
        # Extract intersection nodes
        logger.info(f"Extracting intersection nodes from street network...")
        
        coords = []
        
        # If we have crossing features, use their endpoints as primary coordinates
        if crossings is not None and len(crossings) > 0:
            logger.info("Using crossing features as primary source...")
            
            # Get coordinates from crossing geometries
            seen_coords = set()
            for idx, row in crossings.iterrows():
                geom = row.geometry
                if geom.geom_type == 'LineString':
                    # Get both start and end points of crossing
                    for point in [geom.coords[0], geom.coords[-1]]:
                        lon, lat = point[0], point[1]
                        coord_tuple = (round(lon, 6), round(lat, 6))
                        if coord_tuple not in seen_coords:
                            seen_coords.add(coord_tuple)
                            coords.append({
                                'index': len(coords),
                                'x': lon,
                                'y': lat,
                                'source': 'crossing_edge'
                            })
                elif geom.geom_type == 'Point':
                    # Single point crossing
                    lon, lat = geom.x, geom.y
                    coord_tuple = (round(lon, 6), round(lat, 6))
                    if coord_tuple not in seen_coords:
                        seen_coords.add(coord_tuple)
                        coords.append({
                            'index': len(coords),
                            'x': lon,
                            'y': lat,
                            'source': 'crossing_node'
                        })
        
        # Filter out crossing features that are not near streets
        # This removes railroad crossings, footbridges, etc.
        logger.info("Filtering crossing features to only street-adjacent crossings...")
        filtered_coords = []
        
        for coord in coords:
            # Check if this coordinate is within ~30m of a street node
            is_near_street = False
            for node, data in G.nodes(data=True):
                if 'y' in data and 'x' in data:
                    node_lon, node_lat = data['x'], data['y']
                    # Quick distance check (rough approximation)
                    dist = ((coord['x'] - node_lon)**2 + (coord['y'] - node_lat)**2)**0.5
                    # At Sacramento's latitude, ~0.0003 degrees ≈ 30m
                    if dist < 0.0003:
                        is_near_street = True
                        break
            
            if is_near_street:
                filtered_coords.append(coord)
        
        coords = filtered_coords
        logger.info(f"After filtering to street-adjacent crossings: {len(coords)} coordinates")
        logger.info("Using only crossing features (not adding nearby intersections)")
        
        logger.info(f"Generated {len(coords)} coordinates")
        
        # Save to CSV
        current_file_path = pathlib.Path(__file__)
        current_directory_path = current_file_path.parent.resolve()
        output_dir = Path(current_directory_path) / 'coordinates'
        output_dir.mkdir(exist_ok=True)
        
        df = pd.DataFrame(coords)
        output_file = output_dir / 'sacramento_coords.csv'
        df.to_csv(output_file, index=False)
        
        logger.info(f"Saved to {output_file}")
        logger.info(f"\nSacramento Crossing Stats:")
        logger.info(f"  Total coordinates: {len(coords)}")
        logger.info(f"  By source: {df['source'].value_counts().to_dict()}")
        logger.info(f"  Bounding box: ({df['x'].min():.4f}, {df['y'].min():.4f}) to ({df['x'].max():.4f}, {df['y'].max():.4f})")
        
        return output_file
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    generate_sacramento_crossing_coords()