# Report: Single-Event 3D Visualization with Detector Layers

## Objective

The purpose of this analysis is to visualize a single HGCAL event in 3D while highlighting the detector layers. Each energy deposition (“rechit”) is represented as a point, colored according to its energy, and semi-transparent planes indicate the positions of detector layers. This approach allows a clear understanding of the **shower development** and energy distribution across different layers.

---

## Data

The dataset contains the following key components:

- Number of hits per event.
- 3D coordinates (x, y, z) for each rechit.
- Rechit energy in units of MIP.
- True event energy in GeV.

Each event may contain thousands of hits, with varying energy deposition across layers. The visualization focuses on one event at a time to efficiently manage memory.

---

## Methodology

1. **Configuration and File Loading**  
   - The input file location and output figure directory are specified in a configuration file.  
   - Only the data corresponding to the selected event is loaded into memory to handle large datasets efficiently.

2. **Event Extraction**  
   - The start and end indices for the rechits belonging to the chosen event are determined.  
   - Rechit coordinates and energies for the event are extracted, along with the true event energy.

3. **3D Visualization**  
   - Rechits are plotted as 3D markers with a color scale representing energy.  
   - Hovering over a marker displays its energy.  
   - Unique z-values are identified and semi-transparent planes are drawn to indicate detector layers.

4. **Output**  
   - The 3D figure is displayed interactively, allowing rotation, zoom, and inspection of individual hits.  
   - The figure is also saved as an HTML file for offline viewing and reporting.

---

## Advantages

- **Memory Efficient:** Only the relevant event data is loaded.  
- **Layer Context:** Semi-transparent planes show detector geometry and layer structure.  
- **Interactive Exploration:** Users can inspect individual hits in 3D.  
- **Self-Contained Visualization:** Does not rely on precomputed summaries or external scripts.

---

## Results

The visualization provides a clear depiction of the shower:

- The spatial distribution of hits shows how energy spreads across layers.  
- Energy variations are immediately visible via the color mapping.  
- Layer planes allow understanding of shower depth and energy deposition patterns.  

The true energy of the event is displayed in the figure title for reference.

---

## Notes

- The visualization technique is applicable to any event within the dataset.  
- Marker size and transparency can be tuned for clarity.  
- While CPU is sufficient for plotting, GPU-based processing can be incorporated for additional computation if needed.  
- This method facilitates quality checks, event exploration, and presentation of results in reports or publications.

