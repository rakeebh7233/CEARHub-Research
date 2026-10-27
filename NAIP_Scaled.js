// ============================================================
// NAIP impervious change - SCALE-UP (v5) - west Chatham corridor
// Gross NEW impervious only, 2013 vs 2023 (lost impervious not analyzed).
//
// Drawn imports needed at the top of the Code Editor:
//   studyArea  (polygon)          - west Chatham corridor (Pooler, Port Wentworth,
//                                   Garden City, Bloomingdale + the unincorporated
//                                   land between them). Clipped to the county below.
//   impervious (points, class = 1)
//   pervious   (points, class = 0)
//   stable     (polygons)         - areas you know did not change
//
// Uploaded asset needed from MODE 'SAMPLE' on:
//   NWI wetlands for Chatham (shapefile upload -> table asset)
//
// RUN ORDER (change MODE each time, then press Run):
//   1. 'CHECK'  - point counts, acquisition dates, out-of-bag error, preview.
//   2. 'EXPORT' - start the Tasks-tab export of the change map to an Asset.
//                 Wait for it to finish (can take an hour or more).
//   3. 'SAMPLE' - water/marsh mask, area stats, and the random accuracy sample
//                 (three more Tasks: stats CSV, labeling CSV + key, points asset).
//   4. 'LABEL'  - side-by-side 2013/2023 viewer that steps through the
//                 accuracy points so you can hand-label them.
// If you change training points after step 2, redo steps 2-4.
// ============================================================

// ---------- SETTINGS (edit these) ----------
var MODE = 'CHECK';          // 'CHECK' | 'EXPORT' | 'SAMPLE' | 'LABEL'

var YEAR_OLD = 2013;
var YEAR_NEW = 2023;
var SHADOW_MAX = 60;         // mean R,G,B below this = shadow (either year)
var MIN_PATCH = 50;          // drop new-impervious patches smaller than this (pixels = m2)
var CRS = 'EPSG:26917';      // UTM 17N, all measurements on a 1 m grid

// Your Earth Engine asset folder (create it in the Assets tab first)
var ASSET_ROOT   = 'projects/YOUR-CLOUD-PROJECT/assets/cear';
var CHANGE_ASSET = ASSET_ROOT + '/naip_change_west_2013_2023';
var SAMPLE_ASSET = ASSET_ROOT + '/accuracy_points_west';
var NWI_ASSET    = ASSET_ROOT + '/nwi_chatham';
var NWI_FIELD    = 'ATTRIBUTE';  // NWI code field, e.g. 'E2EM1P'. Check with print(nwi.first()).

// Tidal mask. Masks ONLY estuarine/marine (E*, M*) and tidal riverine (R1*).
// Freshwater wetlands, ponds and lakes stay IN (a filled pond is real new pavement).
var USE_TIDAL_MASK = true;
// Also mask tidal FRESHWATER marsh (palustrine with tidal water regime R/S/T/V).
// Relevant along the Savannah River by Port Wentworth (old rice-field marshes).
var MASK_TIDAL_FRESHWATER = true;

// Accuracy sample: points per stratum
//   1 = mapped NEW impervious
//   2 = impervious in 2023 but NOT mapped new (catches 2013 mistakes and
//       patches removed by the small-patch cleanup)
//   3 = not impervious in 2023, within ~30-60 m of mapped new (missed edges)
//   4 = everything else
var SAMPLE_SIZES = [100, 50, 75, 100];
var SEED = 2026;

// ---------- COMMON: study area and imagery ----------
var chatham = ee.FeatureCollection('TIGER/2018/Counties')
  .filter(ee.Filter.and(ee.Filter.eq('STATEFP', '13'),
                        ee.Filter.eq('NAME', 'Chatham')));
var region = ee.FeatureCollection(studyArea).geometry()
  .intersection(chatham.geometry(), ee.ErrorMargin(1));

var naip = ee.ImageCollection('USDA/NAIP/DOQQ').filterBounds(region);
var rgb = {bands: ['R', 'G', 'B'], min: 0, max: 255};

function tilesFor(year) {
  return naip.filter(ee.Filter.calendarRange(year, year, 'year'));
}
function prep(year) {
  var img = tilesFor(year).mosaic().clip(region);
  var ndvi = img.normalizedDifference(['N', 'R']).rename('NDVI');
  return img.select(['R', 'G', 'B', 'N']).addBands(ndvi);
}

// Only points inside the study area are used.
var training = impervious.merge(pervious).filterBounds(region);

// One random forest per year, each trained on THAT year's pixel values
// at the same points (so camera/brightness differences between years are
// mostly absorbed).
function trainRF(img) {
  var samples = img.sampleRegions({
    collection: training, properties: ['class'], scale: 1, projection: CRS
  });
  var rf = ee.Classifier.smileRandomForest(100)
    .train(samples, 'class', img.bandNames());
  return {samples: samples, rf: rf};
}

function buildChange() {
  var imgOld = prep(YEAR_OLD);
  var imgNew = prep(YEAR_NEW);
  var tOld = trainRF(imgOld);
  var tNew = trainRF(imgNew);

  var impOld = imgOld.classify(tOld.rf).rename('imp2013');
  var impNew = imgNew.classify(tNew.rf).rename('imp2023');

  function brightness(img) {
    return img.select(['R', 'G', 'B']).reduce(ee.Reducer.mean());
  }
  var shadow = brightness(imgOld).lt(SHADOW_MAX)
    .or(brightness(imgNew).lt(SHADOW_MAX)).rename('shadow');

  var newRaw = impNew.eq(1).and(impOld.eq(0)).and(shadow.not());
  var m = newRaw.reproject({crs: CRS, scale: 1}).selfMask();
  var newClean = m.updateMask(m.connectedPixelCount(MIN_PATCH, true).gte(MIN_PATCH));

  // Only keep pixels where both years have imagery
  var dataMask = imgOld.select('R').mask().and(imgNew.select('R').mask());

  var out = ee.Image.cat([
    impOld, impNew, shadow, newClean.unmask(0).rename('new')
  ]).toByte().updateMask(dataMask).clip(region);

  return {out: out, imgOld: imgOld, imgNew: imgNew, tOld: tOld, tNew: tNew};
}

// Tidal mask image: keep = 1 outside tidal NWI polygons, 0 inside.
function tidalKeep() {
  if (!USE_TIDAL_MASK) return {keep: ee.Image(1), tidal: null};
  var nwi = ee.FeatureCollection(NWI_ASSET).filterBounds(region);
  var flagged = nwi.map(function (f) {
    var a = ee.String(f.get(NWI_FIELD));
    var estMar = a.match('^(E|M|R1)').length().gt(0);
    var tidalFresh = a.match('^P[A-Z]{2}[0-9]*(/[A-Z]{2}[0-9]*)?[RSTV]').length().gt(0);
    var flag = MASK_TIDAL_FRESHWATER ? estMar.or(tidalFresh) : estMar;
    return f.set('tidal', flag);
  });
  var tidal = flagged.filter(ee.Filter.eq('tidal', 1));
  var tidalImg = ee.Image(0).byte().paint(tidal, 1);
  return {keep: tidalImg.eq(0), tidal: tidal};
}

// ============================================================
// MODE: CHECK - run this until the numbers look right, then EXPORT
// ============================================================
if (MODE === 'CHECK') {
  var b = buildChange();

  print('Study area (km2):', region.area(1).divide(1e6));
  print('Training points inside study area, by class (0 = pervious, 1 = impervious):',
        training.aggregate_histogram('class'));
  // print('Samples actually used ' + YEAR_OLD + ' (points lost to missing imagery drop out):',
  //       b.tOld.samples.aggregate_histogram('class'));
  // print('Samples actually used ' + YEAR_NEW + ':',
  //       b.tNew.samples.aggregate_histogram('class'));
  print('Out-of-bag error ' + YEAR_OLD + ' (share of training points the forest gets wrong; want < ~0.05):',
        b.tOld.rf.explain().get('outOfBagErrorEstimate'));
  print('Out-of-bag error ' + YEAR_NEW + ':',
        b.tNew.rf.explain().get('outOfBagErrorEstimate'));

  function dates(year) {
    return tilesFor(year).aggregate_array('system:time_start')
      .map(function (t) { return ee.Date(t).format('YYYY-MM-dd'); }).distinct();
  }
  print('Acquisition dates over study area, ' + YEAR_OLD + ':', dates(YEAR_OLD));
  print('Acquisition dates over study area, ' + YEAR_NEW + ':', dates(YEAR_NEW));

  Map.centerObject(region, 12);
  Map.addLayer(tilesFor(YEAR_OLD).mosaic(), rgb, 'VIEW ' + YEAR_OLD, false);
  Map.addLayer(tilesFor(YEAR_NEW).mosaic(), rgb, 'VIEW ' + YEAR_NEW, false);
  Map.addLayer(ee.Image().paint(region, 0, 2), {palette: ['yellow']}, 'Study area');
  Map.addLayer(b.out.select('imp2023').selfMask(), {palette: ['404040']}, 'Impervious ' + YEAR_NEW, false);
  // Zoom to ~16+ before turning this on; the patch cleanup is computed at 1 m.
  Map.addLayer(b.out.select('new').selfMask(), {palette: ['ff0000']}, 'New impervious (zoom in)', false);
  Map.addLayer(training.filter(ee.Filter.eq('class', 1)), {color: 'red'}, 'Training: impervious', false);
  Map.addLayer(training.filter(ee.Filter.eq('class', 0)), {color: 'green'}, 'Training: pervious', false);
}

// ============================================================
// MODE: EXPORT - writes the change map to an Asset (Tasks tab -> Run)
// Bands: imp2013, imp2023 (1 = impervious), shadow (1 = masked), new (1 = new impervious)
// ============================================================
if (MODE === 'EXPORT') {
  var bx = buildChange();
  Export.image.toAsset({
    image: bx.out,
    description: 'naip_change_west_2013_2023',
    assetId: CHANGE_ASSET,
    region: region,
    crs: CRS,
    scale: 1,
    maxPixels: 1e11,
    pyramidingPolicy: {'.default': 'mode'}   // categorical bands: never average
  });
  print('Go to the Tasks tab and click RUN on naip_change_west_2013_2023.');
}

// ============================================================
// MODE: SAMPLE - mask, area stats, stratified accuracy sample
// Everything is computed from the exported asset, so it doesn't change between runs.
// ============================================================
if (MODE === 'SAMPLE') {
  var ch = ee.Image(CHANGE_ASSET);
  var tk = tidalKeep();
  var keep = tk.keep;
  var newB = ch.select('new');

  // "Near mapped new": any new pixel in a 30 m cell, grown by one cell (~30-60 m)
  var near = newB.reduceResolution({reducer: ee.Reducer.max(), maxPixels: 1024})
    .reproject({crs: CRS, scale: 30})
    .focalMax({radius: 1, kernelType: 'square', units: 'pixels'})
    .reproject({crs: CRS, scale: 30});

  var valid = newB.mask().and(keep).and(ch.select('shadow').eq(0));
  var stratum = ee.Image(4).toByte()
    .where(near.eq(1), 3)
    .where(ch.select('imp2023').eq(1), 2)
    .where(newB.eq(1), 1)
    .updateMask(valid)
    .rename('stratum');

  // --- Area statistics (1 m pixels, so pixel counts = m2) ---
  var hist = ee.Dictionary(stratum.reduceRegion({
    reducer: ee.Reducer.frequencyHistogram(), geometry: region,
    crs: CRS, scale: 1, maxPixels: 1e11, tileScale: 4
  }).get('stratum'));

  var sums = ee.Image.cat([
    newB.eq(1).and(keep).rename('new_m2'),
    newB.eq(1).and(keep.not()).rename('new_inside_tidal_mask_m2'),   // check: should be small
    ch.select('shadow').eq(1).and(keep).rename('shadow_m2'),
    keep.not().rename('tidal_masked_m2')
  ]).reduceRegion({
    reducer: ee.Reducer.sum(), geometry: region,
    crs: CRS, scale: 1, maxPixels: 1e11, tileScale: 4
  });

  var stableGeom = ee.FeatureCollection(stable).filterBounds(region).geometry();
  var stableFalse = newB.reduceRegion({
    reducer: ee.Reducer.mean(), geometry: stableGeom,
    crs: CRS, scale: 1, maxPixels: 1e10
  }).get('new');

  var statsFeat = ee.Feature(null, {
    study_area_m2: region.area(1),
    s1_m2: hist.get('1', 0), s2_m2: hist.get('2', 0),
    s3_m2: hist.get('3', 0), s4_m2: hist.get('4', 0),
    stable_false_new_share: stableFalse
  }).set(sums);

  Export.table.toDrive({
    collection: ee.FeatureCollection([statsFeat]),
    description: 'stats_west_2013_2023',
    fileFormat: 'CSV',
    selectors: ['study_area_m2', 's1_m2', 's2_m2', 's3_m2', 's4_m2', 'new_m2',
                'new_inside_tidal_mask_m2', 'shadow_m2', 'tidal_masked_m2',
                'stable_false_new_share']
  });

  // --- Stratified random sample ---
  var pts = stratum.stratifiedSample({
    numPoints: 0, classBand: 'stratum', region: region,
    classValues: [1, 2, 3, 4], classPoints: SAMPLE_SIZES,
    scale: 1, projection: CRS, seed: SEED, geometries: true,
    tileScale: 4, dropNulls: true
  });

  // Shuffle so you label in random order, then number 0..N-1
  pts = pts.randomColumn('r', SEED).sort('r');
  var ptsList = pts.toList(2000);
  var withId = ee.FeatureCollection(
    ee.List.sequence(0, ptsList.size().subtract(1)).map(function (i) {
      var f = ee.Feature(ptsList.get(i));
      var ll = f.geometry().transform('EPSG:4326', 1).coordinates();
      return f.set({id: ee.Number(i).int(), lon: ll.get(0), lat: ll.get(1),
                    label: '', notes: ''})
              .select(['id', 'stratum', 'lon', 'lat', 'label', 'notes']);
    }));

  Export.table.toAsset({
    collection: withId, description: 'accuracy_points_west', assetId: SAMPLE_ASSET
  });
  // Labeling sheet WITHOUT the stratum, so you label blind
  Export.table.toDrive({
    collection: withId, description: 'labels_west', fileFormat: 'CSV',
    selectors: ['id', 'lon', 'lat', 'label', 'notes']
  });
  // Key file (id -> stratum). Don't open this until labeling is done.
  Export.table.toDrive({
    collection: withId, description: 'key_west', fileFormat: 'CSV',
    selectors: ['id', 'stratum']
  });

  print('Tasks tab: run stats_west_2013_2023, accuracy_points_west, labels_west, key_west.');

  Map.centerObject(region, 12);
  Map.addLayer(ee.Image().paint(region, 0, 2), {palette: ['yellow']}, 'Study area');
  if (tk.tidal) Map.addLayer(tk.tidal, {color: '00ffff'}, 'Tidal mask (NWI)', false);
  Map.addLayer(newB.selfMask(), {palette: ['ff0000']}, 'New impervious', false);
  Map.addLayer(stratum, {min: 1, max: 4, palette: ['ff0000', '404040', 'ffaa00', 'cccccc']},
              'Strata (zoom in)', false);
}

// ============================================================
// MODE: LABEL - step through accuracy points, 2013 left / 2023 right
// Record labels in the labels_west CSV (open it in Google Sheets):
//   1 = truly NEW impervious at the ring center (impervious in 2023, not in 2013)
//   0 = not new (pervious in 2023, or already impervious in 2013)
//   U = can't tell (record why in notes)
// The change map is deliberately NOT shown, so you judge only from the photos.
// ============================================================
if (MODE === 'LABEL') {
  var ap = ee.FeatureCollection(SAMPLE_ASSET);
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
    var f = ee.Feature(ap.filter(ee.Filter.eq('id', idx)).first());
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

  ap.size().evaluate(function (n) { total = n; show(); });
}

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
var t13 = tileInfo(YEAR_OLD);
var t23 = tileInfo(YEAR_NEW);
var april = t23.filter(ee.Filter.eq('date', '2023-04-15'));

print('2023 April area (km2):', april.aggregate_sum('km2'));
print('Training points inside April tiles, by class:',
      training.filterBounds(april.geometry()).aggregate_histogram('class'));

Map.addLayer(april.style({color: 'ff00ff', fillColor: 'ff00ff22'}),
             {}, '2023 April tiles (magenta)');
Map.addLayer(t13.filter(ee.Filter.eq('date', '2013-09-21'))
               .style({color: '00ffff', fillColor: '00000000'}),
             {}, '2013 Sept tiles (cyan)', false);

if (MODE === 'CHECK') {
  function cvWrong(img, label) {
    var s = img.sampleRegions({collection: training, properties: ['class'],
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
    print(label + ' confusion matrix (rows = truth 0/1, cols = predicted 0/1):',
          cv.errorMatrix('class', 'pred').array());
    var wrong = cv.filter(ee.Filter.neq('class', ee.Feature(null).get('x'))) ;
    return cv.map(function (f) {
      return f.set('wrong', ee.Number(f.get('class')).neq(ee.Number(f.get('pred'))));
    }).filter(ee.Filter.eq('wrong', 1));
  }
  var wrong13 = cvWrong(prep(YEAR_OLD), '2013');
  var wrong23 = cvWrong(prep(YEAR_NEW), '2023');
  print('2023 wrong points inside April tiles:',
        wrong23.filterBounds(april.geometry()).size(), 'of', wrong23.size());
  Map.addLayer(wrong13, {color: 'ffff00'}, 'Misclassified 2013 (yellow)');
  Map.addLayer(wrong23, {color: 'ff00ff'}, 'Misclassified 2023 (magenta)');
}
