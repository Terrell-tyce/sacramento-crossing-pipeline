"""
Fetch 200 highest entropy satellite images from Sacramento crossings
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from dask import delayed, compute
from dask.diagnostics import ProgressBar
import dask
from PIL import Image
from scipy.stats import entropy
import psutil

from fetch_usgs_imagery import USGSImageryFetcher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
NUM_TOP_IMAGES = 200
COORDS_FILE = Path('coordinates/sacramento_coords.csv')
CACHE_DIR = Path(r"C:/Users/tenoru/Downloads/cache")
OUTPUT_DIR = Path(r"C:/Users/tenoru/Downloads/high entropy images")


class EntropyImageFetcher:
    """Fetch and filter images by entropy - optimized for 16GB RAM"""

    def __init__(self):
        """Initialize with memory-aware Dask config"""
        # Get system memory info
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        logger.info(f"System Memory: {total_memory_gb:.1f} GB")
        
        # Configure Dask for 16GB system:
        # - Limit workers to 4 (i7-1260P has 12 cores, but we don't need all)
        # - Set memory per worker to 2GB
        # - Use threaded scheduler (better for I/O, less memory overhead than processes)
        dask.config.set({
            'scheduler': 'threads',  # Use threads instead of processes (lower memory)
            'num_workers': 4,  # Limit to 4 concurrent tasks
            'distributed.worker.memory.target': 0.8,  # 80% of 2GB = 1.6GB per worker
            'array.chunk-size': '512 MiB',
        })
        logger.info("Dask configured for threaded scheduler with 4 workers")

    @staticmethod
    def calculate_entropy(img):
        """Calculate Shannon entropy of image"""
        if len(img.shape) == 3:
            gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        else:
            gray = img
        
        hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
        hist = hist / hist.sum()
        ent = entropy(hist)
        return ent
    
    @delayed
    def fetch_and_score(self, lat, lon, zoom=21):
        """
        Fetch image and calculate entropy (delayed task)
        Memory is released after task completes
        """
        cache_file = CACHE_DIR / f'crosswalk_{lat}_{lon}.png'
        
        try:
            if cache_file.exists():
                img = np.array(Image.open(cache_file).convert('RGB'))
            else:
                img = USGSImageryFetcher.get_intersection_tile(lon, lat, zoom=zoom)
                if img is None:
                    return None
                Image.fromarray(img).save(cache_file)
            
            ent = self.calculate_entropy(img)
            
            return {
                'lat': lat,
                'lon': lon,
                'entropy': ent,
                'cache_file': cache_file.name,
            }
            
        except Exception as e:
            logger.error(f"Error processing ({lat}, {lon}): {e}")
            return None
    
    def run(self):
        """Fetch all images, rank by entropy, save top 200"""
        
        # Load coordinates
        logger.info(f"Loading coordinates from {COORDS_FILE}")
        coords_df = pd.read_csv(COORDS_FILE)
        logger.info(f"Loaded {len(coords_df)} coordinates")
        
        # Create output directory
        OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        logger.info(f"Saving to {OUTPUT_DIR}")
        
        # Create delayed tasks
        logger.info(f"Creating {len(coords_df)} delayed tasks...")
        tasks = [
            self.fetch_and_score(row['y'], row['x'])
            for idx, row in coords_df.iterrows()
        ]
        
        # Execute with progress bar
        logger.info("Processing images with Dask (threaded scheduler, 4 workers)...")
        with ProgressBar():
            results = compute(*tasks)
        
        # Filter out None results
        results = [r for r in results if r is not None]
        logger.info(f"Successfully processed {len(results)} images")
        
        if not results:
            logger.error("No images successfully processed!")
            return
        
        # Sort by entropy and keep top 200
        results.sort(key=lambda x: x['entropy'], reverse=True)
        top_results = results[:NUM_TOP_IMAGES]
        
        logger.info(f"Top {NUM_TOP_IMAGES} entropy range: {top_results[-1]['entropy']:.4f} - {top_results[0]['entropy']:.4f}")
        
        # Save top images
        logger.info(f"Saving {len(top_results)} images...")
        import shutil
        for i, result in enumerate(top_results, 1):
            src_file = CACHE_DIR / result['cache_file']
            dst_file = OUTPUT_DIR / f"{i:03d}_ent{result['entropy']:.2f}_{result['lat']:.6f}_{result['lon']:.6f}.png"
            
            if src_file.exists():
                shutil.copy(src_file, dst_file)
            else:
                logger.warning(f"Source file not found: {src_file}")
        
        logger.info(f"Complete! Saved {len(top_results)} images to {OUTPUT_DIR}")


if __name__ == '__main__':
    fetcher = EntropyImageFetcher()
    fetcher.run()