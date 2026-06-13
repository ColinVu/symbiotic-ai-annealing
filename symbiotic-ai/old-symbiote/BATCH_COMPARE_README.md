# Batch Image Comparison Tool

Compare all images in a directory and generate comprehensive visual and CSV results.

## What It Does

1. **Embeds all images** in the test directory (skips images where hands can't be detected)
2. **Caches embeddings** to avoid re-processing unchanged images on subsequent runs
3. **Groups images by item name** (e.g., `item04-1.JPG` and `item04-2.JPG` are grouped as `item04`)
4. **Calculates similarity matrix** between all valid images
5. **Generates visual comparison files** - one per item using **cropped hand images** as thumbnails
6. **Generates CSV file** with complete similarity matrix

## Usage

### Basic Usage

```bash
cd symbiote
python batch_compare.py
```

This will:
- Read images from `../images/image-testing/`
- Save results to `../images/testing-results/`

### Custom Directories

```bash
python batch_compare.py --input-dir /path/to/images --output-dir /path/to/results
```

### Verbose Mode

```bash
python batch_compare.py --verbose
```

Shows detailed progress including which images failed and why.

### Disable Caching

```bash
python batch_compare.py --no-cache
```

Forces re-embedding of all images, ignoring the cache. Useful if you want to regenerate everything from scratch.

## Image Naming Convention

Images should follow the pattern: `itemXX-Y.JPG`
- `itemXX` = item identifier (e.g., `item04`)
- `Y` = instance number (e.g., 1, 2, 3...)

Example: `item04-1.JPG`, `item04-2.JPG`, `item04-3.JPG` are all instances of `item04`

## Output Files

### Visual Comparison Images

For each unique item, creates **one** PNG file showing:
- **Large thumbnail** of the base image (first valid instance of that item) - **shows only the cropped hand**
- **Grid of thumbnails** for **all other images** (including other instances of the same item) - **all show cropped hands**
- **Color-coded labels** with:
  - Filename
  - Cosine similarity score
  - Cosine distance
  - Green = Very similar (would match in inference)
  - Blue = Similar
  - Orange = Somewhat similar
  - Red = Not similar

**Example:** If you have `item04-1.JPG`, `item04-2.JPG`, and `item05-1.JPG`:
- Creates `item04_comparison.png` (using `item04-1.JPG` as base, comparing to all others)
- Creates `item05_comparison.png` (using `item05-1.JPG` as base, comparing to all others)

### CSV File: `similarity_matrix.csv`

Complete matrix of all pairwise similarities:
- Rows = base images
- Columns = comparison images
- Values = cosine similarity (0-1)
- Diagonal = 1.0 (self-similarity)

## Example Workflow

1. **Place test images** in `images/image-testing/` following the naming convention:
   ```
   item01-1.JPG
   item01-2.JPG
   item02-1.JPG
   item03-1.JPG
   item03-2.JPG
   item03-3.JPG
   ```

2. **Run batch comparison**:
   ```bash
   cd symbiote
   python batch_compare.py --verbose
   ```

3. **Check results** in `images/testing-results/`:
   - `<item_name>_comparison.png` - Visual comparison for each unique item
   - `similarity_matrix.csv` - Full similarity data (includes ALL images)

4. **Analyze**:
   - Open visual comparisons to see which items look similar
   - Open CSV in Excel/Google Sheets for numerical analysis
   - Check summary statistics printed at the end

## Performance

- **First run**: ~1-2 minutes for model download + embedding time
- **Embedding speed**: ~10-15 seconds per image
- **Comparison speed**: Very fast (after embedding)
- **10 images**: ~2-3 minutes total (first run)
- **50 images**: ~10-15 minutes total (first run)
- **Subsequent runs**: Much faster! Only new/modified images are re-embedded
  - Example: If you add 5 new images to a set of 50, only those 5 are processed (~1 minute)

### Caching Details

- Embeddings and cropped images are cached in `testing-results/.cache/`
- Cache uses filename + modification time as key
- If you modify an image file, it will be automatically re-embedded
- Cache files: `<imagename>_<hash>.npy` (embedding) and `<imagename>_<hash>_seg.npy` (segmented image)

## Troubleshooting

### "Hand not detected" errors
- Ensure images show clear, visible hands
- Check lighting in images
- Try with known-good images first (from `pick-items/` folder)

### Out of memory
- Process fewer images at a time
- Close other applications
- Consider using a machine with more RAM

### Script hangs
- Press Ctrl+C to cancel safely
- Use `--verbose` flag to see where it's hanging
- Check if it's during hand segmentation (slowest part)

## Understanding the Results

### Similarity Scores
- **1.0** = Identical images
- **0.95-1.0** = Very similar (would MATCH in inference)
- **0.85-0.95** = Similar but below threshold
- **0.70-0.85** = Somewhat similar
- **< 0.70** = Not similar

### Distance Scores
- **0.0** = Identical
- **0.0-0.05** = Would MATCH in inference (DETECT_THRESHOLD)
- **0.05-0.15** = Similar but below threshold
- **0.15-0.30** = Somewhat similar
- **> 0.30** = Not similar

## Tips

1. **Start with a small test set** (5-10 images) to verify it works
2. **Use consistent image quality** for best results
3. **Check the summary statistics** to understand overall similarity
4. **Sort CSV by similarity** to find most/least similar pairs
5. **Use visual comparisons** to quickly spot problematic similarities
6. **Leverage caching** - add new images incrementally and only those will be processed
7. **Clear cache** if needed - delete the `.cache` folder or use `--no-cache` flag
8. **Thumbnails show cropped hands** - this is exactly what the model sees during comparison

## Integration with one_on_one.py

This tool uses the same embedding pipeline as `one_on_one.py`, so results should be consistent. Use `one_on_one.py` for:
- Quick spot checks between two images
- Debugging specific comparisons
- Verifying batch results

Use `batch_compare.py` for:
- Comprehensive testing of many images
- Finding similar items across the dataset
- Building similarity matrices for analysis
