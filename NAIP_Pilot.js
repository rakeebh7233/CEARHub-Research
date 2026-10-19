// ============================================================
// NAIP impervious change pilot (v4) - Chatham County, GA
// v4 = shadow mask only (no "pavement is permanent" rule).
//      Both new and lost impervious are counted, with shadows excluded.
//      Optional small-patch cleanup (USE_CLEANUP).
// IMPORTANT: keep your drawn imports (pilot, impervious, pervious, stable)
// at the top of the Code Editor when you paste this in.
// ============================================================

// ---------- SETTINGS (edit these) ----------
var YEAR_OLD = 2013;
var YEAR_NEW = 2023;
var RUN_PILOT = true;

var SHADOW_MAX = 60;     // mean R,G,B below this (0-255) = shadow. Tune with Inspector.
var USE_CLEANUP = true;  // drop small change patches (edge/alignment noise)
var MIN_PATCH = 50;      // patch size threshold in pixels (~m2)

// Placeholder pilot area (keep commented out; you drew your own "pilot")
// var pilot = ee.Geometry.Rectangle([-81.29, 32.09, -81.21, 32.14]);

var CRS = 'EPSG:26917';  // UTM zone 17N, used for all 1 m measurements

// ---------- PART A: NAIP over Chatham ----------
var chatham = ee.FeatureCollection('TIGER/2018/Counties')
  .filter(ee.Filter.and(ee.Filter.eq('STATEFP', '13'),
                        ee.Filter.eq('NAME', 'Chatham')));

var naip = ee.ImageCollection('USDA/NAIP/DOQQ').filterBounds(chatham);

Map.centerObject(pilot, 15);
Map.addLayer(chatham.style({color: 'red', fillColor: '00000000'}), {}, 'Chatham County', false);
Map.addLayer(ee.Image().paint(pilot, 0, 2), {palette: ['yellow']}, 'Pilot area (outline)');

// ---------- Tile collections for the two years ----------
function tilesFor(year) {
  return naip.filter(ee.Filter.calendarRange(year, year, 'year'))
    .filterBounds(pilot);
}
var tilesOld = tilesFor(YEAR_OLD);
var tilesNew = tilesFor(YEAR_NEW);

// Season check (run once, then you can comment these out)
print('Acquisition dates ' + YEAR_OLD + ':',
      tilesOld.aggregate_array('system:time_start')
        .map(function (t) { return ee.Date(t).format('YYYY-MM-dd'); }).distinct());
print('Acquisition dates ' + YEAR_NEW + ':',
      tilesNew.aggregate_array('system:time_start')
        .map(function (t) { return ee.Date(t).format('YYYY-MM-dd'); }).distinct());

var rgb = {bands: ['R', 'G', 'B'], min: 0, max: 255};

// ---------- RAW layers (reference) ----------
Map.addLayer(tilesOld.mosaic(), rgb, 'RAW ' + YEAR_OLD, false);
Map.addLayer(tilesNew.mosaic(), rgb, 'RAW ' + YEAR_NEW, false);

// ---------- Processed imagery ----------
function naipYear(tiles) {
  var img = tiles.mosaic().clip(pilot);
  var ndvi = img.normalizedDifference(['N', 'R']).rename('NDVI');
  return img.select(['R', 'G', 'B', 'N']).addBands(ndvi);
}

var imgOld = naipYear(tilesOld);
var imgNew = naipYear(tilesNew);
Map.addLayer(imgOld, rgb, 'NAIP ' + YEAR_OLD + ' (clipped)', false);
Map.addLayer(imgNew, rgb, 'NAIP ' + YEAR_NEW + ' (clipped)', false);

// ---------- PART B: classify both years and compare ----------
if (RUN_PILOT) {
  var training = impervious.merge(pervious);

  function classify(img) {
    var samples = img.sampleRegions({
      collection: training,
      properties: ['class'],
      scale: 1,
      projection: CRS
    });
    var rf = ee.Classifier.smileRandomForest(100)
      .train(samples, 'class', img.bandNames());
    return img.classify(rf);   // 1 = impervious, 0 = not
  }

  var impOld = classify(imgOld);
  var impNew = classify(imgNew);

  // --- Shadow mask: very dark in either year ---
  function brightness(img) {
    return img.select(['R', 'G', 'B']).reduce(ee.Reducer.mean());
  }
  var shadow = brightness(imgOld).lt(SHADOW_MAX)
    .or(brightness(imgNew).lt(SHADOW_MAX));
  var notShadow = shadow.not();

  // --- Change, excluding shadow pixels ---
  var newRaw = impNew.eq(1).and(impOld.eq(0)).and(notShadow);   // gained impervious
  var lostRaw = impOld.eq(1).and(impNew.eq(0)).and(notShadow);  // lost impervious

  // --- Optional cleanup: drop small change patches ---
  function keepBig(mask) {
    var m = mask.reproject({crs: CRS, scale: 1}).selfMask();
    return m.updateMask(m.connectedPixelCount(MIN_PATCH, true).gte(MIN_PATCH));
  }
  var newImp = USE_CLEANUP ? keepBig(newRaw) : newRaw.selfMask();
  var lostImp = USE_CLEANUP ? keepBig(lostRaw) : lostRaw.selfMask();

  // 0/1 versions for statistics
  var new01 = newImp.unmask(0).clip(pilot);
  var lost01 = lostImp.unmask(0).clip(pilot);
  var changed01 = new01.or(lost01);

  // --- Layers ---
  Map.addLayer(impOld.selfMask(), {palette: ['808080']}, 'Impervious ' + YEAR_OLD, false);
  Map.addLayer(impNew.selfMask(), {palette: ['404040']}, 'Impervious ' + YEAR_NEW, false);
  Map.addLayer(shadow.selfMask(), {palette: ['aa00ff']}, 'Shadow mask (purple)', false);
  Map.addLayer(newImp, {palette: ['ff0000']}, 'New impervious (red)');
  Map.addLayer(lostImp, {palette: ['00aaff']}, 'Lost impervious (blue)');

  // --- Numbers ---
  function meanOver(img, geom) {
    return img.reduceRegion({
      reducer: ee.Reducer.mean(), geometry: geom,
      crs: CRS, scale: 1, maxPixels: 1e10
    });
  }
  function areaOver(img01, geom) {
    return img01.multiply(ee.Image.pixelArea()).reduceRegion({
      reducer: ee.Reducer.sum(), geometry: geom,
      crs: CRS, scale: 1, maxPixels: 1e10
    });
  }

  print('Share of STABLE area flagged as changed, either direction (want small):', meanOver(changed01, stable));
  print('  of which new (red):', meanOver(new01, stable));
  print('  of which lost (blue):', meanOver(lost01, stable));
  print('Share of PILOT area flagged as changed:', meanOver(changed01, pilot));
  print('Share of pilot area masked as shadow:', meanOver(shadow, pilot));
  print('New impervious area in pilot (m2; divide by 4,047 for acres):', areaOver(new01, pilot));
  print('Lost impervious area in pilot (m2):', areaOver(lost01, pilot));
}