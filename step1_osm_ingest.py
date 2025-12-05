"""
Step 1: OSM Ingest - Generate perpendicular crossing lines as seeds for grow-cut
Adapted from official osm_ingest.py to run locally without Modal
Uses the paper's comprehensive crossing detection approach
"""

import geopandas as gpd
import osmnx as ox
import pandas as pd
from pathlib import Path
from shapely.geometry import LineString, Point
import logging
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
PLACE = "Sacramento, California, USA"
OUTPUT_DIR = Path(r"C:/Users/tenoru/Downloads/output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
RADIUS = 15  # meters - only include crossings near major streets

def is_crossing(edge_data):
    """
    Comprehensive crossing detection following the paper's approach.
    Returns True if edge is tagged as a pedestrian crossing in any way.
    
    From paper's osm_ingest.py lines 36-60, checks for:
    - highway=crossing or highway=pedestrian
    - footway=crossing
    - Various crossing subtypes (zebra, marked, signals, island, etc.)
    """
    
    # Direct highway and footway checks
    if edge_data.get("highway") in ["crossing", "pedestrian"]:
        return True
    
    if edge_data.get("footway") == "crossing":
        return True
    
    # Specific crossing types (can have these tags)
    crossing_type_tags = [
        "crossing:raised",
        "crossing:speed_table",
        "crossing:hump",
        "crossing:zebra",
        "crossing:marked",
        "crossing:traffic_signals",
        "crossing:school",
        "crossing:island",
        "crossing:refuge_island",
        "crossing:island:central",
        "crossing:central_island",
        "crossing:island:central:traffic_signals",
        "crossing:island:central:marked",
        "crossing:island:central:zebra",
        "crossing:island:central:uncontrolled",
        "crossing:island:central:unmarked",
        "crossing:unmarked",
        "highway:crossing",
    ]
    
    for tag in crossing_type_tags:
        if tag in edge_data:
            return True
    
    # Check for "pedestrian" tag (catch-all)
    if "pedestrian" in edge_data:
        return True
    
    return False


def map_nodes_to_closest_edges(nodes_gdf, edges_gdf):
    """
    Match each crossing node to the closest road edge.
    Returns dict: {node_index -> closest_edge_index}
    """
    mapping = {}
    for node_idx, node_row in nodes_gdf.iterrows():
        node_geom = node_row.geometry
        
        # Find closest edge
        distances = edges_gdf.geometry.distance(node_geom)
        closest_edge_idx = distances.idxmin()
        
        mapping[node_idx] = closest_edge_idx
    
    return mapping


def create_perpendicular_lines(nodes, edges, mapping, crossing_width=20):
    """
    For each crossing node, create a perpendicular line across the nearest street edge.
    
    Args:
        nodes: GeoDataFrame of crossing nodes (Point geometries)
        edges: GeoDataFrame of street edges (LineString geometries)
        mapping: Dict mapping node indices to edge indices
        crossing_width: Width of generated crossing line in meters (default 20m)
    
    Returns:
        GeoDataFrame with perpendicular LineString geometries
    """
    
    perpendicular_lines = []
    
    for node_idx, edge_idx in mapping.items():
        try:
            node_geom = nodes.loc[node_idx].geometry
            edge_geom = edges.loc[edge_idx].geometry
            
            # Find closest point on edge to the node
            closest_point_on_edge = edge_geom.interpolate(edge_geom.project(node_geom))
            
            # Get the direction vector of the edge
            edge_coords = list(edge_geom.coords)
            
            # Find which segment of the edge is closest
            min_dist = float('inf')
            closest_segment = None
            for i in range(len(edge_coords) - 1):
                p1 = Point(edge_coords[i])
                p2 = Point(edge_coords[i + 1])
                segment = LineString([p1, p2])
                dist = node_geom.distance(segment)
                if dist < min_dist:
                    min_dist = dist
                    closest_segment = (edge_coords[i], edge_coords[i + 1])
            
            if closest_segment is None:
                continue
            
            # Calculate bearing (direction) of the edge
            x1, y1 = closest_segment[0]
            x2, y2 = closest_segment[1]
            dx, dy = x2 - x1, y2 - y1
            bearing = math.atan2(dy, dx)
            
            # Perpendicular bearing (rotate 90 degrees)
            perp_bearing = bearing + math.pi / 2
            
            # Generate perpendicular line endpoints
            half_width = crossing_width / 2
            
            p1_x = closest_point_on_edge.x - half_width * math.cos(perp_bearing)
            p1_y = closest_point_on_edge.y - half_width * math.sin(perp_bearing)
            
            p2_x = closest_point_on_edge.x + half_width * math.cos(perp_bearing)
            p2_y = closest_point_on_edge.y + half_width * math.sin(perp_bearing)
            
            perpendicular_line = LineString([(p1_x, p1_y), (p2_x, p2_y)])
            perpendicular_lines.append({
                'original_node': node_idx,
                'geometry': perpendicular_line,
                'edge_mapping': edge_idx
            })
        
        except Exception as e:
            logger.debug(f"Error creating perpendicular line for node {node_idx}: {e}")
            continue
    
    if not perpendicular_lines:
        return gpd.GeoDataFrame(geometry=[], crs=nodes.crs)
    
    return gpd.GeoDataFrame(perpendicular_lines, crs=nodes.crs)


def osm_ingest_sacramento():
    """
    Main OSM ingest function - generates crossing LineStrings as seeds for grow-cut
    Uses comprehensive crossing detection following the paper's approach
    """
    
    logger.info(f"Starting OSM ingest for {PLACE}...")
    logger.info("Using comprehensive crossing detection (paper's approach)")
    
    # Step 1: Get Sacramento boundary
    logger.info("Fetching Sacramento boundary from OSM...")
    sacramento = ox.geocode_to_gdf(PLACE)
    sacramento_geom = sacramento.geometry.iloc[0]
    
    # Step 2: Get street network for driving (to find crossing nodes near roads)
    logger.info("Downloading Sacramento street network (drive)...")
    
    highway_subtypes = [
        "primary", "primary_link", "secondary", "secondary_link",
        "tertiary", "tertiary_link", "residential", "unclassified",
        "service", "trunk", "trunk_link"
    ]
    
    custom_filter = f'["highway"~"{"|".join(highway_subtypes)}"]["access"!~"private"]'
    
    try:
        G_drive = ox.graph_from_polygon(
            sacramento_geom,
            network_type='drive',
            custom_filter=custom_filter,
            simplify=True
        )
        logger.info(f"Downloaded drive network with {len(G_drive.nodes)} nodes")
    except Exception as e:
        logger.error(f"Error downloading drive network: {e}")
        return None
    
    # Convert to GeoDataFrame
    _, drive_edges = ox.graph_to_gdfs(G_drive)
    drive_edges = drive_edges.to_crs("EPSG:3857")
    
    # Step 3: Get walking network to find crossing edges
    logger.info("Downloading Sacramento street network (walk)...")
    
    try:
        G_walk = ox.graph_from_place(
            PLACE,
            network_type='all',
            simplify=False,
            retain_all=True
        )
        logger.info(f"Downloaded walk network with {len(G_walk.nodes)} nodes")
    except Exception as e:
        logger.error(f"Error downloading walk network: {e}")
        return None
    
    # Step 4: Extract crossing edges using comprehensive approach
    logger.info("Extracting crossing edges from walk network using comprehensive detection...")
    
    crossing_edges_data = []
    
    for u, v, k, d in G_walk.edges(keys=True, data=True):
        if is_crossing(d):
            # Get the geometry
            if 'geometry' in d:
                geom = d['geometry']
            else:
                # Create geometry from node coordinates
                u_node = G_walk.nodes[u]
                v_node = G_walk.nodes[v]
                geom = LineString([(u_node['x'], u_node['y']), (v_node['x'], v_node['y'])])
            
            osmid = d.get('osmid', f"{u}-{v}-{k}")
            
            crossing_edges_data.append({
                'osmid': osmid,
                'geometry': geom,
                'source': 'osm_walk_edge'
            })
    
    if crossing_edges_data:
        crossing_edges_gdf = gpd.GeoDataFrame(crossing_edges_data, crs="EPSG:4326")
        crossing_edges_gdf = crossing_edges_gdf.to_crs("EPSG:3857")
        logger.info(f"Found {len(crossing_edges_gdf)} crossing edges from walk network")
    else:
        crossing_edges_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")
        logger.warning("No crossing edges found in walk network")
    
    # Step 5: Get crossing features (nodes and ways) from OSM features API
    logger.info("Extracting crossing features (nodes and ways) from OSM...")
    
    crossing_tags = {
        'highway': ['crossing', 'pedestrian'],
    }
    
    try:
        crossing_features = ox.features_from_polygon(
            sacramento_geom,
            tags=crossing_tags
        )
        
        # Get crossing nodes (points)
        crossing_nodes_gdf = crossing_features[
            crossing_features.geometry.geom_type == 'Point'
        ].reset_index(drop=True)
        crossing_nodes_gdf = crossing_nodes_gdf.to_crs("EPSG:3857")
        
        logger.info(f"Found {len(crossing_nodes_gdf)} crossing nodes from OSM features")
        
    except Exception as e:
        logger.warning(f"Error extracting crossing features: {e}")
        crossing_nodes_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")
    
    # Step 6: Generate perpendicular lines from crossing nodes
    logger.info("Generating perpendicular crossing lines from nodes...")
    
    if len(crossing_nodes_gdf) > 0:
        # Map nodes to closest edges
        mapping = map_nodes_to_closest_edges(crossing_nodes_gdf, drive_edges)
        
        # Create perpendicular lines
        artificial_crossings = create_perpendicular_lines(
            nodes=crossing_nodes_gdf,
            edges=drive_edges,
            mapping=mapping,
            crossing_width=20  # 20m crossing width
        )
        
        logger.info(f"Generated {len(artificial_crossings)} artificial crossing lines from nodes")
    else:
        artificial_crossings = gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")
        logger.warning("No crossing nodes found to generate artificial crossings")
    
    # Step 7: Combine all crossing edges (existing + generated)
    logger.info("Combining all crossing sources...")
    
    all_crossings_list = []
    
    if len(crossing_edges_gdf) > 0:
        all_crossings_list.append(crossing_edges_gdf[['geometry']].assign(source='osm_edge'))
    
    if len(artificial_crossings) > 0:
        all_crossings_list.append(artificial_crossings[['geometry']].assign(source='generated_from_node'))
    
    if all_crossings_list:
        all_crossings = pd.concat(all_crossings_list, ignore_index=True)
    else:
        logger.error("No crossing edges or nodes found!")
        return None
    
    # Remove duplicates based on geometry
    all_crossings = all_crossings.drop_duplicates(subset=['geometry'])
    
    logger.info(f"Total unique crossings after combining sources: {len(all_crossings)}")
    
    # Step 8: Save as Shapefile for grow-cut step
    logger.info(f"Saving {len(all_crossings)} crossing edges...")
    
    # Convert back to EPSG:4326 for storage
    all_crossings_4326 = all_crossings.to_crs("EPSG:4326")
    
    output_shp = OUTPUT_DIR / "crosswalk_edges_v2_0.shp"
    all_crossings_4326.to_file(output_shp, index=False)
    logger.info(f"Saved crossing edges to {output_shp}")
    
    # Also save as GeoJSON for reference
    output_geojson = OUTPUT_DIR / "crossing_edges.geojson"
    all_crossings_4326.to_file(output_geojson, driver='GeoJSON')
    logger.info(f"Saved crossing edges to {output_geojson}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("OSM INGEST SUMMARY")
    logger.info("="*60)
    logger.info(f"Total crossing edges: {len(all_crossings)}")
    if 'source' in all_crossings.columns:
        logger.info(f"Sources:")
        logger.info(all_crossings['source'].value_counts().to_string())
    logger.info("="*60)
    
    return all_crossings_4326


if __name__ == "__main__":
    osm_ingest_sacramento()