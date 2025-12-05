"""
Local Dask-based processing pipeline for Sacramento crossings
No cloud services, runs on your machine with all CPU cores
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse
import os
from dask import delayed, compute
from dask.diagnostics import ProgressBar
import dask
from PIL import Image

from fetch_usgs_imagery import USGSImageryFetcher
import osmnx as ox

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure Dask
dask.config.set({
    'distributed.scheduler.work-stealing': True,
    'array.chunk-size': '512 MiB',
})


class SacramentoProcessor:
    """Main processing pipeline"""

    # Required columns in the coordinates CSV
    REQUIRED_COLUMNS = ['x', 'y']

    def __init__(self, coords_file='coordinates/sacramento_coords.csv', 
                 output_dir='outputs', cache_dir=r"M:\Intersection-Images\cache"):
        """
        Args:
            coords_file: Path to sacramento_coords.csv
            output_dir: Where to save results
            cache_dir: Where to cache downloaded images (default: './cache')
        """
        self.coords_file = Path(coords_file)
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        print(self.cache_dir)
        # Create directories if they don't exist
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Using Coordinates File: {self.coords_file}")
        logger.info(f"Saving Images to Cache: {self.cache_dir}")
        logger.info(f"Saving Results to: {self.output_dir}")
    
    def load_coordinates(self):
        """Load intersection coordinates from CSV with validation"""
        logger.info(f"Loading coordinates from {self.coords_file}")
        
        if not self.coords_file.exists():
            raise FileNotFoundError(f"Coordinates file not found: {self.coords_file}")
        
        df = pd.read_csv(self.coords_file)
        
        # Validate required columns exist
        missing_columns = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"CSV is missing required columns: {missing_columns}. "
                f"Found columns: {list(df.columns)}"
            )
        
        logger.info(f"Loaded {len(df)} intersection coordinates")
        logger.info(f"Columns present: {list(df.columns)}")
        return df
    
    def load_crossing_edges(self):
        """Load crossing edges from OpenStreetMap"""
        logger.info("Loading crossing edges from OpenStreetMap...")
        
        try:
            sacramento = ox.geocode_to_gdf("Sacramento, California, USA")
            
            # Get all pedestrian crossing ways
            crossing_tags = {'footway': 'crossing', 'highway': 'crossing'}
            crossings = ox.features_from_polygon(
                sacramento.geometry.iloc[0],
                tags=crossing_tags
            )
            
            logger.info(f"Found {len(crossings)} crossing ways/nodes from OSM")
            return crossings
            
        except Exception as e:
            logger.error(f"Error loading crossing edges: {e}")
            return None
    
    @delayed
    def process_intersection(self, idx, lon, lat, zoom=21):
        """
        Process a single intersection:
        1. Check cache for image.
        2. Fetch and save if not cached.
        3. [Next steps: segment, analyze, etc.]
        
        Args:
            zoom: Target zoom level (higher = more zoomed in, default 20).
                  The fetcher will try zoom ± 1 if the target fails.
        """
        
        cache_file = self.cache_dir / f'index_{idx}.png'
        
        try:
            # --- STAGE 1: Caching and Download ---
            if cache_file.exists():
                # If cached, load it and skip the slow network call
                img = np.array(Image.open(cache_file).convert('RGB'))
                source = 'cache'
            else:
                # If not cached, download the image
                img = USGSImageryFetcher.get_intersection_tile(lon, lat, zoom=zoom)
                source = 'download'
                
                if img is None:
                    return {
                        'idx': idx, 'lon': lon, 'lat': lat, 
                        'status': 'failed', 'error': 'Could not fetch imagery'
                    }
                
                # SAVE the newly downloaded image to the cache
                Image.fromarray(img).save(cache_file)
                
            # --- STAGE 2: Processing (Future Implementation) ---
            # TODO: Segment image with SAM model
            # TODO: Apply grow-cut algorithm
            
            return {
                'idx': idx, 'lon': lon, 'lat': lat, 
                'status': 'success', 
                'source': source,  # Tells you if it was downloaded or loaded from cache
                'imagery_shape': img.shape,
            }
            
        except Exception as e:
            logger.error(f"Error processing intersection {idx}: {e}")
            return {
                'idx': idx, 'lon': lon, 'lat': lat, 
                'status': 'error', 'error': str(e)
            }
    
    def run_pipeline(self, sample_size=None):
        """
        Run full processing pipeline using Dask
        
        Args:
            sample_size: Number of intersections to process (None = all)
        """
        
        # Load data
        coords_df = self.load_coordinates()
        crossings = self.load_crossing_edges()
        
        # Sample if requested
        if sample_size and sample_size < len(coords_df):
            logger.info(f"Sampling {sample_size} intersections from {len(coords_df)}")
            coords_df = coords_df.sample(n=sample_size, random_state=42)
        
        logger.info(f"Processing {len(coords_df)} intersections with Dask...")
        
        # Create delayed tasks
        tasks = [
            self.process_intersection(idx, row['x'], row['y'])
            for idx, row in coords_df.iterrows()
        ]
        
        # Execute with progress bar
        logger.info(f"Starting processing on {dask.system.CPU_COUNT} cores...")
        with ProgressBar():
            results = compute(*tasks)
        
        # Collect results
        results_df = pd.DataFrame(results)
        
        # Save results
        output_file = self.output_dir / 'sacramento_crossings_results.csv'
        results_df.to_csv(output_file, index=False)
        logger.info(f"Saved results to {output_file}")
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("PROCESSING SUMMARY")
        logger.info("="*60)
        logger.info(f"Total intersections: {len(results_df)}")
        logger.info(f"Successful: {len(results_df[results_df['status']=='success'])}")
        logger.info(f"Failed: {len(results_df[results_df['status']=='failed'])}")
        logger.info(f"Errors: {len(results_df[results_df['status']=='error'])}")
        logger.info("="*60)
        
        return results_df


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description='Process Sacramento crossings with Dask')
    parser.add_argument('--coords', type=str, default='coordinates/sacramento_coords.csv',
                       help='Path to coordinates CSV file')
    parser.add_argument('--output', type=str, default=r'C:/Users/tenoru/Downloads/output',
                       help='Output directory for results')
    parser.add_argument('--cache', type=str, default=r"C:/Users/tenoru/Downloads/cache",
                       help='Cache directory for downloaded images')
    parser.add_argument('--sample', type=int, default=None, 
                       help='Number of intersections to sample (None = all)')
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("SACRAMENTO CROSSING ANALYSIS - DASK LOCAL PROCESSING")
    logger.info("="*60)
    logger.info(f"CPU Cores Available: {dask.system.CPU_COUNT}")
    
    try:
        # Initialize processor
        processor = SacramentoProcessor(
            coords_file=args.coords,
            output_dir=args.output,
            cache_dir=args.cache
        )
        
        # Run pipeline
        processor.run_pipeline(sample_size=args.sample)
        
        logger.info("Processing complete!")
        
    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())