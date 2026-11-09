// ============================================================
// NAIP impervious change - v6 (map version v3) - west Chatham corridor
// v6 = 3 classes per year (0 vegetation/water, 1 impervious, 2 bare/cleared),
//      separate bare-ground training points per year, a texture band
//      (pavement is smooth, dirt and dry grass are mottled), and an EVAL mode
//      that scores the new map against the EXISTING 325 labeled points.
//
// Drawn imports needed (set each one's type to FeatureCollection with the gear icon):
//   studyArea  (polygon)
//   impervious (points)  - impervious in BOTH years        -> class 1
//   pervious   (points)  - vegetation/water in BOTH years  -> class 0
//   bare13     (points)  - bare/cleared ground in 2013     -> class 2 (2013 model only)
//   bare23     (points)  - bare/cleared ground in 2023     -> class 2 (2023 model only)
//   stable     (polygons) - areas you know did not change
//   exclude    (polygons) - disposal area, mine pit
// Class values are set by the script, so the 'class' property on the imports
// doesn't matter.
//
// RUN ORDER:
//   1. 'CHECK'  - counts, out-of-bag error, 3-class cross-validation, preview.
//   2. 'EXPORT' - change map to Asset (v3). Wait for it to finish.
//   3. 'VIEW'   - browse v3 (fast).
//   4. 'EVAL'   - two Tasks: eval_v3_west (map values at your 325 labeled points)
//                 and stats_v3_west (v3 areas). Then run accuracy_v3.py.
//   'LABEL'     - same 2013/2023 viewer as before (jump to any accuracy point id).
// Do NOT rerun the old SAMPLE mode: your labels belong to the existing sample.
// ============================================================

// ---------- SETTINGS ----------
var MODE = 'CHECK';          // 'CHECK' | 'EXPORT' | 'VIEW' | 'EVAL' | 'LABEL'
var SHOW_DIAGNOSTICS = true; // CHECK mode: April tiles + cross-validation (slow)

var YEAR_OLD = 2013;
var YEAR_NEW = 2023;
var SHADOW_MAX = 60;         // mean R,G,B below this = shadow (either year)
var MIN_PATCH = 50;          // drop new patches smaller than this (m2)
var TEX_RADIUS = 2;          // texture window = (2*2+1) = 5 x 5 m
var CRS = 'EPSG:26917';

var ASSET_ROOT   = 'projects/cearhub/assets';
var CHANGE_ASSET = ASSET_ROOT + '/naip_change_west_2013_2023_v3';
var SAMPLE_ASSET = ASSET_ROOT + '/accuracy_points_west';   // existing labeled sample
var NWI_ASSET    = ASSET_ROOT + '/nwi_chatham';
var NWI_FIELD    = 'ATTRIBUTE';

var USE_TIDAL_MASK = true;
var MASK_TIDAL_FRESHWATER = true;

// Class colors: 0 vegetation/water, 1 impervious, 2 bare
var CLASS_VIS = {min: 0, max: 2, palette: ['33aa33', '404040', 'e0c080']};

// ---------- STUDY AREA AND IMAGERY ----------
var chatham = ee.FeatureCollection('TIGER/2018/Counties')
  .filter(ee.Filter.and(ee.Filter.eq('STATEFP', '13'),
                        ee.Filter.eq('NAME', 'Chatham')));
var region = ee.Geometry(ee.FeatureCollection(studyArea).geometry()
  .intersection(chatham.geometry(), ee.ErrorMargin(1))
  .difference(ee.FeatureCollection(exclude).geometry(), ee.ErrorMargin(1))
  .getInfo());

var naip = ee.ImageCollection('USDA/NAIP/DOQQ').filterBounds(region);
var rgb = {bands: ['R', 'G', 'B'], min: 0, max: 255};

function tilesFor(year) {
  return naip.filter(ee.Filter.calendarRange(year, year, 'year'));
}
function clipIfExport(img) {
  return MODE === 'EXPORT' ? img.clip(region) : img;
}

function addStatic(fc, vis, name, shown) {
  fc.map(function (f) {
    return f.setGeometry(f.geometry().transform('EPSG:4326', 0.1));
  }).evaluate(function (geo, err) {
    if (err) { print('Layer "' + name + '" failed:', err); return; }
    var feats = geo.features.map(function (f) {
      return ee.Feature(ee.Geometry(f.geometry), f.properties);
    });
    Map.addLayer(ee.FeatureCollection(feats), vis, name, shown);
  });
}

// Bands: R, G, B, N, NDVI, NDWI, TEX.
// TEX = standard deviation of brightness in a 5 x 5 m window, computed on a fixed
// 1 m grid so the window is the same size in both years (2013 and 2023 NAIP have
// different native pixel sizes). Because of that fixed grid, live map layers that
// use the classifier only work zoomed in (~16+).
function prep(year) {
  var img = clipIfExport(tilesFor(year).mosaic());
  var ndvi = img.normalizedDifference(['N', 'R']).rename('NDVI');
  var ndwi = img.normalizedDifference(['G', 'N']).rename('NDWI');
  var bright = img.select(['R', 'G', 'B']).reduce(ee.Reducer.mean())
    .reproject({crs: CRS, scale: 1});
  var tex = bright.reduceNeighborhood({
    reducer: ee.Reducer.stdDev(),
    kernel: ee.Kernel.square(TEX_RADIUS, 'pixels')
  }).rename('TEX');
  return img.select(['R', 'G', 'B', 'N']).addBands(ndvi).addBands(ndwi).addBands(tex);
}

// ---------- TRAINING DATA ----------
function asClass(fc, c) {
  return ee.FeatureCollection(fc).map(function (f) { return f.set('class', c); });
}
var impPts  = asClass(impervious, 1);
var pervPts = asClass(pervious, 0);
// Each year's model gets the both-years points plus that year's bare points.
var training13 = impPts.merge(pervPts).merge(asClass(bare13, 2)).filterBounds(region);
var training23 = impPts.merge(pervPts).merge(asClass(bare23, 2)).filterBounds(region);

function trainRF(img, coll) {
  var samples = img.sampleRegions({
    collection: coll, properties: ['class'], scale: 1, projection: CRS
  });
  var rf = ee.Classifier.smileRandomForest(100)
    .train(samples, 'class', img.bandNames());
  return {samples: samples, rf: rf};
}

function cleanPatches(mask) {
  var m = mask.reproject({crs: CRS, scale: 1}).selfMask();
  return m.updateMask(m.connectedPixelCount(MIN_PATCH, true).gte(MIN_PATCH));
}

function buildChange() {
  var imgOld = prep(YEAR_OLD);
  var imgNew = prep(YEAR_NEW);
  var tOld = trainRF(imgOld, training13);
  var tNew = trainRF(imgNew, training23);

  var c13 = imgOld.classify(tOld.rf).rename('cls2013');
  var c23 = imgNew.classify(tNew.rf).rename('cls2023');

  function brightness(img) {
    return img.select(['R', 'G', 'B']).reduce(ee.Reducer.mean());
  }
  var shadow = brightness(imgOld).lt(SHADOW_MAX)
    .or(brightness(imgNew).lt(SHADOW_MAX)).rename('shadow');

  // New impervious: impervious in 2023, NOT impervious in 2013 (2013 may be
  // vegetation OR bare - bare in 2013 then paved by 2023 counts as new).
  var newImp = cleanPatches(c23.eq(1).and(c13.neq(1)).and(shadow.not()));
  // New bare/cleared: bare in 2023, vegetation/water in 2013.
  var newBare = cleanPatches(c23.eq(2).and(c13.eq(0)).and(shadow.not()));

  var dataMask = imgOld.select('R').mask().and(imgNew.select('R').mask());

  var out = clipIfExport(ee.Image.cat([
    c13.eq(1).rename('imp2013'),
    c23.eq(1).rename('imp2023'),
    shadow,
    newImp.unmask(0).rename('new'),
    newBare.unmask(0).rename('newBare'),
    c13, c23
  ]).toByte().updateMask(dataMask));

  return {out: out, imgOld: imgOld, imgNew: imgNew, tOld: tOld, tNew: tNew};
}

// Tidal mask (corrected regex for mixed classes like PFO1/4R, PSS1/2T).
function tidalKeep() {
  if (!USE_TIDAL_MASK) return {keep: ee.Image(1), tidal: null};
  var nwi = ee.FeatureCollection(NWI_ASSET).filterBounds(region);
  var flagged = nwi.map(function (f) {
    var a = ee.String(ee.Algorithms.If(f.get(NWI_FIELD), f.get(NWI_FIELD), ''));
    var estMar = a.match('^(E|M|R1)').length().gt(0);
    var tidalFresh = a.match('^P[A-Z]{2}[0-9]*(/([A-Z]{2}[0-9]*|[0-9]+))?[RSTV]')
      .length().gt(0);
    var flag = MASK_TIDAL_FRESHWATER ? estMar.or(tidalFresh) : estMar;
    return f.set('tidal', flag);
  });
  var tidal = flagged.filter(ee.Filter.eq('tidal', 1));
  var tidalImg = ee.Image(0).byte().paint(tidal, 1);
  return {keep: tidalImg.eq(0), tidal: tidal};
}

// ============================================================
// MODE: CHECK
// ============================================================
if (MODE === 'CHECK') {
  var b = buildChange();

  print('Study area (km2):', region.area(1).divide(1e6));
  print('2013 training points by class (0 veg/water, 1 impervious, 2 bare):',
        training13.aggregate_histogram('class'));
  print('2023 training points by class:', training23.aggregate_histogram('class'));
  print('If class 2 shows 1 instead of ~40 / ~60, set bare13/bare23 import type to FeatureCollection.');
  print('Out-of-bag error ' + YEAR_OLD + ':', b.tOld.rf.explain().get('outOfBagErrorEstimate'));
  print('Out-of-bag error ' + YEAR_NEW + ':', b.tNew.rf.explain().get('outOfBagErrorEstimate'));

  Map.centerObject(region, 12);
  Map.addLayer(tilesFor(YEAR_OLD).mosaic(), rgb, 'VIEW ' + YEAR_OLD, false);
  Map.addLayer(tilesFor(YEAR_NEW).mosaic(), rgb, 'VIEW ' + YEAR_NEW, true);
  Map.addLayer(ee.Image().paint(region, 0, 2), {palette: ['yellow']}, 'Study area');
  // Slow, live classifier layers. Zoom to ~16+ before turning on.
  Map.addLayer(b.out.select('cls2023'), CLASS_VIS, 'Classes ' + YEAR_NEW + ' (slow)', false);
  Map.addLayer(b.out.select('cls2013'), CLASS_VIS, 'Classes ' + YEAR_OLD + ' (slow)', false);
  Map.addLayer(b.out.select('new').selfMask(), {palette: ['ff0000']}, 'New impervious (slow)', false);
  Map.addLayer(b.out.select('newBare').selfMask(), {palette: ['ff9900']}, 'New bare (slow)', false);

  if (SHOW_DIAGNOSTICS) {
    var t23 = tileInfo(YEAR_NEW);
    var april = t23.filter(ee.Filter.eq('date', '2023-04-15'));
    print('Training points inside April 2023 tiles, by class:',
          training23.filterBounds(april.geometry()).aggregate_histogram('class'));
    addStatic(april.select(['date']), {color: 'ff00ff', fillColor: 'ff00ff22'},
              '2023 April tiles (magenta)', false);

    var wrong13 = cvWrong(prep(YEAR_OLD), training13, '2013');
    var wrong23 = cvWrong(prep(YEAR_NEW), training23, '2023');
    addStatic(wrong13.select(['class', 'pred']), {color: 'ffff00'}, 'Misclassified 2013 (yellow)', true);
    addStatic(wrong23.select(['class', 'pred']), {color: 'ff00ff'}, 'Misclassified 2023 (magenta)', true);
    // The error that matters most: not-impervious called impervious.
    addStatic(wrong23.filter(ee.Filter.and(ee.Filter.neq('class', 1), ee.Filter.eq('pred', 1)))
                .select(['class', 'pred']),
              {color: 'ff8800'}, '2023 veg/bare called impervious (orange)', true);
  }
}

// ============================================================
// MODE: VIEW
// ============================================================
if (MODE === 'VIEW') {
  var cv_ = ee.Image(CHANGE_ASSET);
  Map.centerObject(region, 12);
  Map.addLayer(tilesFor(YEAR_OLD).mosaic(), rgb, 'VIEW ' + YEAR_OLD, false);
  Map.addLayer(tilesFor(YEAR_NEW).mosaic(), rgb, 'VIEW ' + YEAR_NEW, true);
  Map.addLayer(ee.Image().paint(region, 0, 2), {palette: ['yellow']}, 'Study area');
  Map.addLayer(cv_.select('cls2023'), CLASS_VIS, 'Classes ' + YEAR_NEW, false);
  Map.addLayer(cv_.select('cls2013'), CLASS_VIS, 'Classes ' + YEAR_OLD, false);
  Map.addLayer(cv_.select('shadow').selfMask(), {palette: ['aa00ff']}, 'Shadow mask (purple)', false);
  Map.addLayer(cv_.select('newBare').selfMask(), {palette: ['ff9900']}, 'New bare/cleared (orange)', false);
  Map.addLayer(cv_.select('new').selfMask(), {palette: ['ff0000']}, 'New impervious (red)', true);
}

// ============================================================
// MODE: EXPORT
// Bands: imp2013, imp2023, shadow, new, newBare, cls2013, cls2023
// ============================================================
if (MODE === 'EXPORT') {
  var bx = buildChange();
  Export.image.toAsset({
    image: bx.out,
    description: 'naip_change_west_2013_2023_v3',
    assetId: CHANGE_ASSET,
    region: region,
    crs: CRS,
    scale: 1,
    maxPixels: 1e11,
    pyramidingPolicy: {'.default': 'mode'}
  });
  print('Tasks tab: RUN naip_change_west_2013_2023_v3.');
}

// ============================================================
// MODE: EVAL - score v3 against the existing labeled points
// ============================================================
if (MODE === 'EVAL') {
  var ch = ee.Image(CHANGE_ASSET);
  var ap = ee.FeatureCollection(SAMPLE_ASSET);

  var vals = ch.select(['new', 'newBare', 'cls2013', 'cls2023']).sampleRegions({
    collection: ap, properties: ['id'], scale: 1, projection: CRS
  });
  Export.table.toDrive({
    collection: vals, description: 'eval_v3_west', fileFormat: 'CSV',
    selectors: ['id', 'new', 'newBare', 'cls2013', 'cls2023']
  });

  var keep = tidalKeep().keep;
  var ok = ch.select('shadow').eq(0).and(keep);
  var sums = ee.Image.cat([
    ch.select('new').eq(1).and(ok).rename('new_m2'),
    ch.select('newBare').eq(1).and(ok).rename('newbare_m2')
  ]).reduceRegion({
    reducer: ee.Reducer.sum(), geometry: region,
    crs: CRS, scale: 1, maxPixels: 1e11, tileScale: 4
  });
  var stableGeom = ee.FeatureCollection(stable).filterBounds(region).geometry();
  var stableFalse = ch.select('new').reduceRegion({
    reducer: ee.Reducer.mean(), geometry: stableGeom,
    crs: CRS, scale: 1, maxPixels: 1e10
  }).get('new');
  Export.table.toDrive({
    collection: ee.FeatureCollection([ee.Feature(null, sums)
      .set('stable_false_new_share', stableFalse)]),
    description: 'stats_v3_west', fileFormat: 'CSV',
    selectors: ['new_m2', 'newbare_m2', 'stable_false_new_share']
  });

  print('Tasks tab: run eval_v3_west and stats_v3_west.');
  Map.centerObject(region, 12);
  Map.addLayer(ch.select('new').selfMask(), {palette: ['ff0000']}, 'New impervious v3', false);
}

// ============================================================
// MODE: LABEL - unchanged 2013 / 2023 viewer for the accuracy points
// ============================================================
if (MODE === 'LABEL') {
  var apL = ee.FeatureCollection(SAMPLE_ASSET);
  var viewOld = tilesFor(YEAR_OLD).mosaic();
  var viewNew = tilesFor(YEAR_NEW).mosaic();

  var leftMap = ui.Map();
  var rightMap = ui.Map();
  leftMap.addLayer(viewOld, rgb, String(YEAR_OLD));
  rightMap.addLayer(viewNew, rgb, String(YEAR_NEW));
  leftMap.add(ui.Label(String(YEAR_OLD), {position: 'bottom-left', fontWeight: 'bold'}));
  rightMap.add(ui.Label(String(YEAR_NEW), {position: 'bottom-right', fontWeight: 'bold'}));
  ui.Map.Linker([leftMap, rightMap]);
  ui.root.widgets().reset([ui.SplitPanel({
    firstPanel: leftMap, secondPanel: rightMap, orientation: 'horizontal', wipe: false
  })]);

  var idx = 0;
  var total = null;
  var info = ui.Label('Loading points...');

  function setRing(map, layer) {
    if (map.layers().length() > 1) map.layers().set(1, layer);
    else map.layers().add(layer);
  }
  function show() {
    var f = ee.Feature(apL.filter(ee.Filter.eq('id', idx)).first());
    var ring = ee.Image().byte().paint(
      ee.FeatureCollection([ee.Feature(f.geometry().buffer(2.5))]), 1, 2);
    setRing(leftMap, ui.Map.Layer(ring, {palette: ['ffff00']}, 'point'));
    setRing(rightMap, ui.Map.Layer(ring, {palette: ['ffff00']}, 'point'));
    leftMap.centerObject(f.geometry(), 20);
    f.toDictionary(['lon', 'lat']).evaluate(function (d) {
      info.setValue('Point id ' + idx + ' of ' + (total - 1) +
                    '  (' + d.lat.toFixed(6) + ', ' + d.lon.toFixed(6) + ')');
    });
  }
  var prevBtn = ui.Button('< Prev', function () { if (idx > 0) { idx--; show(); } });
  var nextBtn = ui.Button('Next >', function () {
    if (total === null || idx < total - 1) { idx++; show(); }
  });
  var jump = ui.Textbox({placeholder: 'jump to id', onChange: function (v) {
    var n = parseInt(v, 10);
    if (!isNaN(n) && n >= 0 && (total === null || n < total)) { idx = n; show(); }
  }});
  leftMap.add(ui.Panel(
    [info, ui.Panel([prevBtn, nextBtn], ui.Panel.Layout.flow('horizontal')), jump],
    null, {position: 'top-left', width: '260px'}));
  apL.size().evaluate(function (n) { total = n; show(); });
}

// ============================================================
// Helpers
// ============================================================
function tileInfo(year) {
  var list = tilesFor(year).toList(100);
  return ee.FeatureCollection(list.map(function (img) {
    img = ee.Image(img);
    var g = img.geometry().intersection(region, ee.ErrorMargin(1));
    return ee.Feature(g, {
      date: img.date().format('YYYY-MM-dd'),
      km2: g.area(1).divide(1e6)
    });
  }));
}

// 5-fold cross-validation for one year. Prints a 3 x 3 confusion matrix
// (rows = truth 0/1/2, columns = predicted 0/1/2) and returns the wrong points.
function cvWrong(img, coll, label) {
  var s = img.sampleRegions({collection: coll, properties: ['class'],
                             scale: 1, projection: CRS, geometries: true})
             .randomColumn('r', 7);
  var parts = ee.List.sequence(0, 4).map(function (k) {
    k = ee.Number(k);
    var lo = k.multiply(0.2), hi = k.add(1).multiply(0.2);
    var test = s.filter(ee.Filter.and(ee.Filter.gte('r', lo), ee.Filter.lt('r', hi)));
    var train = s.filter(ee.Filter.or(ee.Filter.lt('r', lo), ee.Filter.gte('r', hi)));
    var rf = ee.Classifier.smileRandomForest(100)
      .train(train, 'class', img.bandNames());
    return test.classify(rf, 'pred');
  });
  var cv = ee.FeatureCollection(parts).flatten();
  print(label + ' confusion matrix (rows = truth 0 veg / 1 imp / 2 bare, cols = predicted):',
        cv.errorMatrix('class', 'pred', [0, 1, 2]).array());
  return cv.map(function (f) {
    return f.set('wrong', ee.Number(f.get('class')).neq(ee.Number(f.get('pred'))));
  }).filter(ee.Filter.eq('wrong', 1));
}