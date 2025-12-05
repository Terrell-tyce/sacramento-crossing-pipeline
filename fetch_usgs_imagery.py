"""Fetch free satellite imagery using OpenStreetMap/Mapbox tiles"""
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import logging
import time
from pathlib import Path
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class USGSImageryFetcher:
    """Fetch satellite imagery from free tile services"""
    
    # Tile servers (multiple options for fallback)
    # Using Mapbox free satellite tiles as primary
    TILE_SERVERS = [
        "https://a.tiles.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}.jpg70",
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "https://mt0.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",  # Google satellite
    ]
    
    # Tile size in pixels
    TILE_SIZE = 256

    @classmethod
    def _get_tile_by_xyz(cls, xtile, ytile, zoom, server_idx=0):
        """
        Internal function to fetch a tile directly by its XYZ coordinates.
        Tries multiple servers if one fails.
        """
        for attempt, tile_server in enumerate(cls.TILE_SERVERS[server_idx:], server_idx):
            try:
                # Format URL based on server pattern
                url = tile_server.format(x=xtile, y=ytile, z=zoom)
                
                # Fetch with timeout and user agent
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, timeout=10, headers=headers)

                if response.status_code == 200:
                    # Convert to numpy array
                    img = Image.open(BytesIO(response.content))
                    img_array = np.array(img)

                    # Ensure RGB (not RGBA)
                    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
                        img_array = img_array[:, :, :3]
                    elif len(img_array.shape) == 2:  # Grayscale
                        img_array = np.stack([img_array]*3, axis=-1)

                    return img_array
                else:
                    logger.debug(f"Server {attempt} failed: HTTP {response.status_code}")
                    
            except Exception as e:
                logger.debug(f"Server {attempt} error: {e}")
                continue
        
        logger.warning(f"Failed to fetch tile at {zoom}/{ytile}/{xtile} from all servers")
        return None
    
    @classmethod
    def get_tile(cls, lon, lat, zoom=18):
        """
        Fetch a satellite imagery tile by lon/lat coordinates.
        
        Args:
            lon: Longitude
            lat: Latitude
            zoom: Zoom level
            
        Returns:
            numpy array (H, W, 3) RGB image, or None if failed
        """
        
        try:
            # Convert lat/lon to tile coordinates
            xtile = cls._lon_to_tile(lon, zoom)
            ytile = cls._lat_to_tile(lat, zoom)
            
            return cls._get_tile_by_xyz(xtile, ytile, zoom)
                
        except Exception as e:
            logger.error(f"Error fetching imagery for ({lon}, {lat}): {e}")
            return None
        
    @staticmethod
    def _lon_to_tile(lon, zoom):
        """Convert longitude to web mercator tile X coordinate"""
        return int((lon + 180) / 360 * (2 ** zoom))
    
    @staticmethod
    def _lat_to_tile(lat, zoom):
        """Convert latitude to web mercator tile Y coordinate"""
        lat_rad = math.radians(lat)
        n = 2 ** zoom
        ytile = (1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n
        return int(ytile)
    
    @staticmethod
    def _tile_to_lon(xtile, zoom):
        """Convert tile X coordinate to longitude (top-left of tile)"""
        n = 2.0 ** zoom
        lon = xtile / n * 360.0 - 180.0
        return lon

    @staticmethod
    def _tile_to_lat(ytile, zoom):
        """Convert tile Y coordinate to latitude (top-left of tile)"""
        n = 2.0 ** zoom
        lat_rad = np.arctan(np.sinh(np.pi * (1 - 2 * ytile / n)))
        lat = np.degrees(lat_rad)
        return lat

    @classmethod
    def get_intersection_tile(cls, lon, lat, size_pixels=1629, zoom=21):
        """
        Fetch satellite imagery centered on intersection by STITCHING multiple tiles.
        Includes fallback to lower zoom levels if tiles are unavailable.
        
        Args:
            lon: Longitude
            lat: Latitude
            size_pixels: Output image size (default 1629)
            zoom: Target zoom level (default 20). Will try zoom ± 1 if needed.
        """
        
        # Try zoom levels in order of preference (prefer the requested zoom, then adjust)
        zoom_levels = [zoom, zoom - 1, zoom + 1, zoom + 2, zoom - 2]
        
        for current_zoom in zoom_levels:
            logger.info(f"Attempting to fetch tiles at zoom level {current_zoom}...")
            result = cls._fetch_tiles_at_zoom(lon, lat, size_pixels, current_zoom)
            
            if result is not None:
                # Check if result has meaningful content (not all black)
                mean_val = np.mean(result)
                if mean_val > 10:  # Average pixel value > 10
                    logger.info(f"Successfully fetched tiles at zoom {current_zoom}")
                    return result
                else:
                    logger.warning(f"Tiles at zoom {current_zoom} appear mostly blank, trying lower zoom...")
        
        # If all zoom levels fail, return None
        logger.error(f"Could not fetch imagery for ({lon}, {lat}) at any zoom level")
        return None

    @classmethod
    def _fetch_tiles_at_zoom(cls, lon, lat, size_pixels, zoom):
        """
        Internal method to fetch and stitch tiles at a specific zoom level.
        Returns None if too many tiles fail.
        """
        
        # Determine the extent of the area we need
        TILE_COUNT_PER_SIDE = 7  # 7 * 256 = 1792 pixels
        
        # Get the tile index of the central point
        center_xtile = cls._lon_to_tile(lon, zoom)
        center_ytile = cls._lat_to_tile(lat, zoom)
        
        # Calculate the starting tile index (top-left of the 7x7 grid)
        start_xtile = center_xtile - (TILE_COUNT_PER_SIDE // 2)
        start_ytile = center_ytile - (TILE_COUNT_PER_SIDE // 2)
        
        # Download all tiles in the grid
        tile_images = []
        failed_tiles = 0
        
        for j in range(TILE_COUNT_PER_SIDE):
            row_images = []
            for i in range(TILE_COUNT_PER_SIDE):
                xtile = start_xtile + i
                ytile = start_ytile + j
                
                tile = cls._get_tile_by_xyz(xtile, ytile, zoom)
                
                if tile is None:
                    # Fill with medium gray instead of black
                    tile = np.ones((cls.TILE_SIZE, cls.TILE_SIZE, 3), dtype=np.uint8) * 128
                    failed_tiles += 1
                    
                row_images.append(tile)
            tile_images.append(row_images)
        
        # If more than 60% of tiles failed, this zoom level likely doesn't have good coverage
        total_tiles = TILE_COUNT_PER_SIDE * TILE_COUNT_PER_SIDE
        if failed_tiles > (total_tiles * 0.6):
            logger.warning(f"Too many failed tiles at zoom {zoom} ({failed_tiles}/{total_tiles}), trying lower zoom...")
            return None
            
        # Stitch the tiles together
        stitched_image = np.vstack([np.hstack(row) for row in tile_images])
        
        # Crop to the desired size (1629x1629) and center
        center_x = stitched_image.shape[1] // 2
        center_y = stitched_image.shape[0] // 2
        
        start_y = center_y - (size_pixels // 2)
        end_y = start_y + size_pixels
        
        start_x = center_x - (size_pixels // 2)
        end_x = start_x + size_pixels

        cropped_image = stitched_image[
            start_y : end_y,
            start_x : end_x,
            :
        ]
        
        logger.info(f"Successfully stitched and cropped image to {cropped_image.shape} at zoom {zoom}")
        return cropped_image