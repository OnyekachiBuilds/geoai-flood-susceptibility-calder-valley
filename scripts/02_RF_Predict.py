import rasterio
import numpy as np
import joblib

print("Loading trained Random Forest model...")
model = joblib.load("RF_Flood_Model.pkl")

base = r"C:\Users\user\Documents\GeoAI-Based Flood Susceptibility Mapping in Calder Valley Using GIS, Remote Sensing, and Random Forest Machine Learning"

raster_files = {
    "DEM": base + r"\DEM_CalderValley_Final.tif",
    "Slope": base + r"\Slope_Final.tif",
    "NDVI": base + r"\NDVI_Aligned.tif",
    "LandCover": base + r"\LandCover_Aligned.tif",
    "Rainfall": base + r"\Rainfall_Aligned.tif",
    "DistanceRiver": base + r"\DistanceRiver_Final_Proper.tif",
    "FlowAccum": base + r"\FlowAccum_ClippedToDEM.tif"
}

print("Opening raster layers...")
rasters = {name: rasterio.open(path) for name, path in raster_files.items()}

reference = rasters["DEM"]
profile = reference.profile.copy()
rows, cols = reference.shape

arrays = []

print("Reading raster values...")
for name in raster_files.keys():
    arr = rasters[name].read(1).astype("float32")

    nodata = rasters[name].nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan

    arrays.append(arr)
    print(f"{name}: shape={arr.shape}, min={np.nanmin(arr)}, max={np.nanmax(arr)}")

print("Stacking raster layers...")
stack = np.stack(arrays, axis=-1)

X_pred = stack.reshape(-1, len(raster_files))

mask = np.any(np.isnan(X_pred), axis=1)
valid_pixels = X_pred[~mask]

print("Total pixels:", X_pred.shape[0])
print("Valid pixels for prediction:", valid_pixels.shape[0])

print("Predicting flood susceptibility probability...")
probabilities = model.predict_proba(valid_pixels)[:, 1]

result = np.full(X_pred.shape[0], np.nan, dtype="float32")
result[~mask] = probabilities.astype("float32")
result = result.reshape(rows, cols)

profile.update(
    dtype="float32",
    count=1,
    nodata=-9999,
    compress="lzw"
)

output_file = "Flood_Susceptibility_RF.tif"

result_to_write = np.where(np.isnan(result), -9999, result).astype("float32")

print("Writing output raster...")
with rasterio.open(output_file, "w", **profile) as dst:
    dst.write(result_to_write, 1)

for src in rasters.values():
    src.close()

print("Flood susceptibility raster created successfully:")
print(output_file)
print("Output min:", np.nanmin(result))
print("Output max:", np.nanmax(result))
print("Output mean:", np.nanmean(result))