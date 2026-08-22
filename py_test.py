import os
import gc
import json
import rasterio
from rasterio.warp import transform_bounds
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import folium
from pathlib import Path
import subprocess
import shutil

# ==========================================
# БЛОК 1: НАСТРОЙКИ КОНВЕЙЕРА
# ==========================================
# --- Главный рубильник ---
RUN_ODM = True                # True - запускать долгую сшивку в Docker, False - использовать готовый tif

# --- Пути к папкам ---
ODM_PROJECT_DIR = "DroneTest"  # Папка проекта ODM (внутри ОБЯЗАТЕЛЬНО должна быть папка 'images' с сырыми фото)
WORK_DIR = "test_fields"       # Папка, где скрипт будет искать/сохранять итоговый .tif для аналитики
OUTPUT_DIR = "output_results"  # Папка для готового HTML-отчета

# --- Имена файлов ---
TIF_FILENAME = "odm1.tif"      # Имя файла, с которым работает математика
PROJECT_NAME = "MyField"       # Имя для финального отчета (например: Map_MyField.html)

RENDER_SCALE = 1               # 1 - максимальное качество, 2+ - сжатие для экономии памяти

# ==========================================
# МОДУЛЬ 1: ПРЕДОБРАБОТКА (СШИВКА ЧЕРЕЗ DOCKER)
# ==========================================
def run_opendronemap(project_dir, output_dir, final_filename):
    """Запускает OpenDroneMap и копирует готовый файл в рабочую директорию"""
    abs_project_dir = os.path.abspath(project_dir)
    print(f"\n--- [ЭТАП 1] Старт конвейера OpenDroneMap ---")
    print(f"Директория проекта: {abs_project_dir}")
    
    docker_cmd = [
        "docker", "run", "-ti", "--rm",
        "-v", f"{abs_project_dir}:/datasets/test",
        "opendronemap/odm",
        "--project-path", "/datasets", "test",
        "--feature-quality", "low",
        "--fast-orthophoto"
    ]
    
    try:
        print("Запуск математического ядра ODM (это займет время, шум кулеров - норма)...")
        subprocess.run(docker_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ОШИБКА] Сбой при работе OpenDroneMap: {e}")
        return False
        
    # Ищем результат в технических папках ODM
    odm_output_path = os.path.join(abs_project_dir, "odm_orthophoto", "odm_orthophoto.tif")
    target_path = os.path.join(output_dir, final_filename)
    
    if os.path.exists(odm_output_path):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        shutil.copy2(odm_output_path, target_path)
        print(f"Файл успешно сшит и скопирован: {target_path}")
        return True
    
    print(f"[ОШИБКА] Итоговый файл не найден по пути: {odm_output_path}")
    return False

# ==========================================
# МОДУЛЬ 2: МАТЕМАТИКА И ГЕНЕРАЦИЯ КАРТЫ
# ==========================================
def generate_field_report(tif_path, report_name, output_folder):
    """Обрабатывает один TIF файл и создает HTML-дашборд"""
    print(f"\n--- [ЭТАП 2] Аналитика и генерация отчета ---")
    print(f"Чтение файла: {tif_path}")
    
    # 1. Читаем границы файла для центрирования карты
    with rasterio.open(tif_path) as src:
        left, bottom, right, top = src.bounds
        min_lon, min_lat, max_lon, max_lat = transform_bounds(src.crs, 'EPSG:4326', left, bottom, right, top)
        
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    
    # 2. Инициализация карты
    m = folium.Map(location=[center_lat, center_lon], tiles=None)
    m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google', name='Google Satellite', max_zoom=24, max_native_zoom=19   
    ).add_to(m)
    
    cmap = plt.cm.RdYlGn
    norm = colors.Normalize(vmin=0.0, vmax=0.9)
    js_tiles_data = []

    # 3. Умное чтение каналов (Мультиспектр vs RGB)
    with rasterio.open(tif_path) as src:
        out_h = int(src.height / RENDER_SCALE)
        out_w = int(src.width / RENDER_SCALE)
        
        band_count = src.count
        print(f"Обнаружено каналов: {band_count}. Выбор алгоритма...")
        
        if band_count >= 5:
            print("-> Используется формула классического NDVI (Инфракрасный датчик)")
            red = src.read(3, out_shape=(out_h, out_w)).astype('float32')
            nir = src.read(5, out_shape=(out_h, out_w)).astype('float32')
            empty_mask = (red == 0) & (nir == 0)
            
            np.seterr(divide='ignore', invalid='ignore')
            denominator = np.add(nir, red) + 1e-8
            ndvi = np.empty_like(red, dtype='float32')
            np.subtract(nir, red, out=ndvi)
            np.divide(ndvi, denominator, out=ndvi)
        else:
            print("-> Используется формула VARI (Обычная RGB камера)")
            red = src.read(1, out_shape=(out_h, out_w)).astype('float32')
            green = src.read(2, out_shape=(out_h, out_w)).astype('float32')
            blue = src.read(3, out_shape=(out_h, out_w)).astype('float32')
            
            if band_count >= 4:
                alpha = src.read(4, out_shape=(out_h, out_w)).astype('float32')
                empty_mask = (alpha == 0)
            else:
                empty_mask = (red == 0) & (green == 0) & (blue == 0)
            
            np.seterr(divide='ignore', invalid='ignore')
            denominator = (green + red - blue) + 1e-8
            vari = (green - red) / denominator
            ndvi = np.clip((vari + 0.3), 0, 1) 
        
        ndvi[empty_mask] = np.nan
    
    # 4. Подготовка данных для JS дашборда
    lookup_scale = max(1, max(ndvi.shape) // 400)
    ndvi_lookup = ndvi[::lookup_scale, ::lookup_scale]
    ndvi_lookup_int = np.where(np.isnan(ndvi_lookup), -999, np.round(ndvi_lookup * 100)).astype(np.int16)
    
    js_tiles_data.append({
        "name": "Field Data",
        "bounds": [[min_lat, min_lon], [max_lat, max_lon]],
        "rows": ndvi_lookup_int.shape[0],
        "cols": ndvi_lookup_int.shape[1],
        "data": ndvi_lookup_int.tolist()
    })
    
    # 5. Отрисовка слоя на карте
    rgba_img = cmap(norm(ndvi))
    rgba_img[np.isnan(ndvi), 3] = 0.0 
    
    folium.raster_layers.ImageOverlay(
        image=rgba_img, 
        bounds=[[min_lat, min_lon], [max_lat, max_lon]], 
        opacity=0.8, 
        name="Field Data"
    ).add_to(m)
    
    del red, ndvi, empty_mask, rgba_img, ndvi_lookup, ndvi_lookup_int
    gc.collect()
        
    folium.LayerControl(position='bottomleft').add_to(m)

    # 6. Инъекция твоего проверенного интерфейса
    custom_ui_script = f"""
    <style>
        #lang-panel {{ position: absolute; top: 15px; right: 15px; z-index: 9999; background: rgba(255, 255, 255, 0.95); padding: 8px 12px; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; backdrop-filter: blur(5px); }}
        #lang-panel select {{ font-size: 15px; border: none; background: transparent; cursor: pointer; outline: none; font-weight: bold; color: #333; }}
        #stats-panel {{ position: absolute; bottom: 25px; right: 15px; z-index: 9999; background: rgba(255, 255, 255, 0.95); padding: 20px; border-radius: 12px; box-shadow: 0 5px 20px rgba(0,0,0,0.3); font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; width: 280px; backdrop-filter: blur(5px); color: #333; }}
        #stats-panel h3 {{ margin: 0 0 15px 0; font-size: 18px; text-align: center; color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 8px; }}
        .stat-row {{ display: flex; justify-content: space-between; margin-top: 12px; font-size: 16px; font-weight: 500; }}
    </style>

    <div id="lang-panel">
        <select id="lang-select" onchange="window.updateLang(this.value)">
            <option value="ru">🇷🇺 Русский</option>
            <option value="en">🇬🇧 English</option>
            <option value="by">🇧🇾 Беларуская</option>
        </select>
    </div>

    <div id="stats-panel">
        <h3 id="ui-stats-title">Аналитика поля</h3>
        <canvas id="stressChart" width="240" height="240"></canvas>
        <div class="stat-row">
            <span id="ui-avg-label">Средний NDVI:</span>
            <span id="avg-ndvi-val" style="font-weight: bold; color: #1a9850;">-</span>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

    <script>
    document.addEventListener("DOMContentLoaded", function() {{
        var map_instance = null;
        for (var key in window) {{
            if (key.startsWith("map_")) {{ map_instance = window[key]; break; }}
        }}
        
        if (map_instance) {{
            var tilesData = {json.dumps(js_tiles_data)};
            
            var translations = {{
                'ru': {{ lat: 'Широта', lon: 'Долгота', val: 'Индекс', rec: 'Статус', alert: 'Критический стресс', warn: 'Требует внимания', norm: 'Оптимальная зона', statsTitle: 'Аналитика поля', avg: 'Средний Индекс', percent: '% от выбранного' }},
                'en': {{ lat: 'Latitude', lon: 'Longitude', val: 'Index', rec: 'Status', alert: 'Critical Stress', warn: 'Needs Attention', norm: 'Optimal Zone', statsTitle: 'Field Analytics', avg: 'Avg Index', percent: '% of selected' }},
                'by': {{ lat: 'Шырата', lon: 'Даўгата', val: 'Індэкс', rec: 'Статус', alert: 'Крытычны стрэс', warn: 'Патрабуе ўвагі', norm: 'Аптымальная зона', statsTitle: 'Аналітыка поля', avg: 'Сярэдні Індэкс', percent: '% ад абранага' }}
            }};

            window.currentLang = 'ru';
            var myChart = null; 
            
            var activeLayers = new Set();
            tilesData.forEach(t => activeLayers.add(t.name));

            window.updateStats = function() {{
                var redCnt = 0, yellowCnt = 0, greenCnt = 0;
                var totalNdvi = 0, validPixels = 0;
                
                tilesData.forEach(t => {{
                    if (activeLayers.has(t.name)) {{
                        for(let r=0; r<t.rows; r++) {{
                            for(let c=0; c<t.cols; c++) {{
                                let v = t.data[r][c];
                                if(v !== -999) {{
                                    validPixels++;
                                    totalNdvi += v;
                                    if(v < 30) redCnt++;
                                    else if(v < 60) yellowCnt++;
                                    else greenCnt++;
                                }}
                            }}
                        }}
                    }}
                }});
                
                var avgNdvi = validPixels > 0 ? (totalNdvi / validPixels / 100).toFixed(2) : "0.00";
                
                var avgColor = "#1a9850";
                if(avgNdvi < 0.3) avgColor = "#d73027";
                else if(avgNdvi < 0.6) avgColor = "#f4b400";
                
                var avgEl = document.getElementById('avg-ndvi-val');
                avgEl.innerText = avgNdvi;
                avgEl.style.color = avgColor;

                if (myChart) {{
                    myChart.data.datasets[0].data = [redCnt, yellowCnt, greenCnt];
                    myChart.update();
                }}
            }};

            map_instance.on('overlayadd', function(e) {{
                activeLayers.add(e.name);
                window.updateStats();
            }});
            map_instance.on('overlayremove', function(e) {{
                activeLayers.delete(e.name);
                window.updateStats();
            }});

            function initChart() {{
                var ctx = document.getElementById('stressChart').getContext('2d');
                var t = translations[window.currentLang];
                
                myChart = new Chart(ctx, {{
                    type: 'doughnut',
                    data: {{
                        labels: [t.alert, t.warn, t.norm],
                        datasets: [{{
                            data: [0, 0, 0],
                            backgroundColor: ['#d73027', '#fee08b', '#1a9850'],
                            borderWidth: 2,
                            borderColor: '#ffffff'
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        cutout: '65%',
                        plugins: {{
                            legend: {{ position: 'bottom', labels: {{ padding: 15, font: {{ family: 'Segoe UI', size: 13 }} }} }},
                            tooltip: {{ callbacks: {{
                                label: function(context) {{
                                    var dataset = context.chart.data.datasets[0];
                                    var total = dataset.data.reduce((a, b) => a + b, 0);
                                    var percent = total > 0 ? ((context.raw / total) * 100).toFixed(1) : 0;
                                    return ' ' + percent + translations[window.currentLang].percent;
                                }}
                            }} }}
                        }}
                    }}
                }});
                
                window.updateStats(); 
            }}

            setTimeout(initChart, 500);

            window.updateLang = function(lang) {{
                window.currentLang = lang;
                var t = translations[lang];
                
                document.getElementById('ui-stats-title').innerText = t.statsTitle;
                document.getElementById('ui-avg-label').innerText = t.avg;
                
                if(myChart) {{
                    myChart.data.labels = [t.alert, t.warn, t.norm];
                    myChart.update();
                }}

                var popupTitleLat = document.getElementById('t-lat');
                if (popupTitleLat) {{
                    document.getElementById('t-lat').innerText = t.lat;
                    document.getElementById('t-lon').innerText = t.lon;
                    document.getElementById('t-val').innerText = t.val;
                    document.getElementById('t-rec').innerText = t.rec;
                    
                    var recEl = document.getElementById('t-rec-val');
                    var statusCode = recEl.getAttribute('data-status');
                    recEl.innerText = t[statusCode];
                }}
            }};

            map_instance.on('click', function(e) {{
                var lat = e.latlng.lat;
                var lng = e.latlng.lng;
                var clickedValue = null;
                
                for(var i = tilesData.length - 1; i >= 0; i--) {{
                    var t = tilesData[i];
                    if (!activeLayers.has(t.name)) continue;
                    
                    var minLat = t.bounds[0][0], minLon = t.bounds[0][1];
                    var maxLat = t.bounds[1][0], maxLon = t.bounds[1][1];
                    
                    if(lat >= minLat && lat <= maxLat && lng >= minLon && lng <= maxLon) {{
                        var rowPercent = (maxLat - lat) / (maxLat - minLat);
                        var colPercent = (lng - minLon) / (maxLon - minLon);
                        
                        var rowIndex = Math.floor(rowPercent * t.rows);
                        var colIndex = Math.floor(colPercent * t.cols);
                        
                        if (rowIndex >= 0 && rowIndex < t.rows && colIndex >= 0 && colIndex < t.cols) {{
                            var val = t.data[rowIndex][colIndex];
                            if (val !== -999) {{
                                clickedValue = val / 100.0;
                                break;
                            }}
                        }}
                    }}
                }}
                
                if (clickedValue === null) return; 
                
                var colorBox, statusCode;
                if (clickedValue < 0.3) {{ colorBox = '#d73027'; statusCode = 'alert'; }}
                else if (clickedValue < 0.6) {{ colorBox = '#fee08b'; statusCode = 'warn'; }}
                else {{ colorBox = '#1a9850'; statusCode = 'norm'; }}
                
                var t = translations[window.currentLang];
                var statusText = t[statusCode];

                var htmlContent = `
                <div style="font-size: 18px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; min-width: 250px; line-height: 1.6; color: #333; padding: 5px;">
                    <div style="margin-bottom: 12px; font-size: 15px; color: #666; border-bottom: 1px dashed #ccc; padding-bottom: 8px;">
                        <b><span id="t-lat">${{t.lat}}</span>:</b> ${{lat.toFixed(6)}}<br>
                        <b><span id="t-lon">${{t.lon}}</span>:</b> ${{lng.toFixed(6)}}
                    </div>
                    <div style="background: #f8f9fa; padding: 12px; border-radius: 8px; border-left: 6px solid ${{colorBox}}; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                        <div style="display: flex; align-items: center; margin-bottom: 8px;">
                            <b style="width: 140px; font-size: 16px;"><span id="t-val">${{t.val}}</span>:</b> 
                            <span style="display:inline-block; width:22px; height:22px; background:${{colorBox}}; border:1px solid rgba(0,0,0,0.2); border-radius: 4px; margin-right: 10px;"></span>
                            <span style="font-size: 22px; font-weight: 900; letter-spacing: 0.5px;">${{clickedValue.toFixed(2)}}</span>
                        </div>
                        <div style="display: flex; align-items: center; font-size: 16px;">
                            <b style="width: 140px;"><span id="t-rec">${{t.rec}}</span>:</b> 
                            <strong id="t-rec-val" data-status="${{statusCode}}" style="color:${{colorBox}}; background: rgba(255,255,255,0.8); padding: 2px 6px; border-radius: 4px;">${{statusText}}</strong>
                        </div>
                    </div>
                </div>`;
                
                L.popup().setLatLng(e.latlng).setContent(htmlContent).openOn(map_instance);
            }});
        }}
    }});
    </script>
    """
    m.get_root().html.add_child(folium.Element(custom_ui_script))
    
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    html_filename = os.path.join(output_folder, f"Map_{report_name}.html")
    m.save(html_filename)
    print(f"Отчет успешно сохранен: {html_filename}\n")

# ==========================================
# ТОЧКА ВХОДА (MAIN)
# ==========================================
if __name__ == "__main__":
    print("\n=== ЗАПУСК АГРО-КОНВЕЙЕРА ===")
    
    target_tif = os.path.join(WORK_DIR, TIF_FILENAME)
    
    # 1. Запуск дрона (если включено)
    if RUN_ODM:
        success = run_opendronemap(ODM_PROJECT_DIR, WORK_DIR, TIF_FILENAME)
        if not success:
            print("Остановка работы из-за ошибки сшивки.")
            exit(1)
    else:
        print(f"[ПРОПУСК] Обработка фото с дрона отключена. Ищем готовый файл {target_tif}")
            
    # 2. Генерация карты
    if os.path.exists(target_tif):
        generate_field_report(target_tif, PROJECT_NAME, OUTPUT_DIR)
        print("Конвейер завершил работу!")
    else:
        print(f"\n[КРИТИЧЕСКАЯ ОШИБКА] Файл для анализа не найден: {target_tif}")
        print(f"Убедитесь, что папка '{WORK_DIR}' существует и содержит файл '{TIF_FILENAME}'.")