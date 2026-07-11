import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Конфигурация API
#API_BASE_URL = "http://localhost:8000" 
st.set_page_config(page_title="Scoring DIS System", layout="wide", page_icon="📊")

# Дефолтные признакки и веса
DEFAULT_FEATURES = [
    {"name": "Конкурент", "weight": 0.25, "type": "Бинарный", "formula": "binary"},
    {"name": "Была угроза отключения", "weight": 0.25, "type": "Бинарный", "formula": "binary"},
    {"name": "Время всех сессий в мин", "weight": 0.20, "type": "Числовой", "formula": "min_max_inverse"},
    {"name": "Срок жизни от регистрации", "weight": 0.15, "type": "Числовой", "formula": "min_max_inverse"},
    {"name": "%Скидки", "weight": 0.10, "type": "Процентный", "formula": "min_max_inverse"},
    {"name": "Количество ОС", "weight": 0.05, "type": "Числовой", "formula": "step_os"}
]

# КОНФИГУРАЦИЯ API
# Заменить на реальный адрес бэкенда после запуска!!!
API_BASE_URL = "http://localhost:8000" 
st.set_page_config(page_title="Churn & Win-back System", layout="wide")
st.title("Система скоринга клиентов ДИС")

FORMULA_OPTIONS = {
    "binary": "Бинарный (0/1)",
    "min_max": "Min-Max Нормализация",
    "min_max_inverse": "Min-Max Инвертированная (меньше = выше риск)",
    "step_os": "Ступенчатая (ОС: 1=30, 2=70, 3+=100)"
}

FEATURE_TYPES = ["Бинарный", "Числовой", "Процентный"]

# Инициализация состояния
if 'scoring_mode' not in st.session_state:
    st.session_state['scoring_mode'] = 'ml' # 'ml' или 'manual'
if 'manual_features' not in st.session_state:
    st.session_state['manual_features'] = DEFAULT_FEATURES.copy()
if 'result_data' not in st.session_state:
    st.session_state['result_data'] = None

# Боковая панель: выбор режима и настройки
with st.sidebar:
    st.header("Настройки системы")
    
    mode = st.radio(
        "Режим скоринга:",
        options=["ML Модель (CatBoost)", "Ручной скоринг"],
        index=0 if st.session_state['scoring_mode'] == 'ml' else 1,
        key="mode_radio"
    )
    
    current_mode = 'ml' if "ML" in mode else 'manual'
    if st.session_state['scoring_mode'] != current_mode:
        st.session_state['scoring_mode'] = current_mode
        st.session_state['result_data'] = None # Сброс результатов при смене режима
    
    st.divider()
    
    # Загрузка файла
    uploaded_file = st.file_uploader("Загрузите Excel файл (.xlsx)", type=["xlsx"])
    
    if uploaded_file is not None:
        st.success(f"Файл выбран")
        
        # Кнопка расчета 
        btn_label = "Рассчитать ML скоринг" if current_mode == 'ml' else "🧮 Рассчитать ручной скоринг"
        
        if st.button(btn_label, type="primary"):
            with st.spinner("Обработка данных..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
                    
                    if current_mode == 'ml':
                        params = {"mode": "churn"} # Или winback, если добавите переключатель
                        endpoint = f"{API_BASE_URL}/score"
                    else:
                        # Валидация весов перед отправкой
                        total_weight = sum(f['weight'] for f in st.session_state['manual_features'])
                        if abs(total_weight - 1.0) > 0.001:
                            st.error(f"Сумма весов должна быть равна 1.0. Текущая сумма: {total_weight:.4f}")
                            st.stop()
                        
                        payload = {
                            "features": st.session_state['manual_features'],
                            "target_column": "final_probability"
                        }
                        files["config"] = (None, str(payload).encode('utf-8'), "application/json")
                        endpoint = f"{API_BASE_URL}/score/manual"
                    
                    response = requests.post(endpoint, files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state['result_data'] = data
                        st.success("Расчет завершен!")
                    else:
                        st.error(f"Ошибка сервера ({response.status_code}): {response.text}")
                        
                except Exception as e:
                    st.error(f"Ошибка соединения: {str(e)}")

# Основной экран
if st.session_state['result_data']:
    data = st.session_state['result_data']
    df_full = pd.DataFrame(data.get('clients', []))
    summary = data.get('summary', {})
    
    # 1. ДАШБОРДЫ (KPI + ГРАФИКИ)
    st.subheader("Дашборд скоринга")
    
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1:
        st.metric("Всего клиентов", summary.get('total_clients', len(df_full)))
    with col_kpi2:
        high_risk = summary.get('high_risk', len(df_full[df_full.get('risk_level') == 'Высокий']))
        st.metric("Высокий риск", high_risk, delta_color="inverse")
    with col_kpi3:
        avg_prob = summary.get('avg_risk', df_full['final_probability'].mean() if 'final_probability' in df_full.columns else 0)
        st.metric("Средняя вероятность ухода", f"{avg_prob:.2%}")
    with col_kpi4:
        avg_completeness = df_full.get('feature_completeness', pd.Series([0]*len(df_full))).mean()
        st.metric("Средняя полнота данных", f"{avg_completeness:.2%}")
    
    # Графики
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        risk_counts = df_full['risk_level'].value_counts().reset_index()
        risk_counts.columns = ['Уровень риска', 'Количество']
        fig_risk = px.pie(risk_counts, values='Количество', names='Уровень риска', 
                          color='Уровень риска',
                          color_discrete_map={'Высокий': '#FF4B4B', 'Средний': '#FFA500', 'Низкий': '#00CC96'})
        fig_risk.update_layout(title_text="Распределение по уровням риска", height=300)
        st.plotly_chart(fig_risk, use_container_width=True)
        
    with chart_col2:
        # Средние баллы по факторам (если есть в данных)
        factor_cols = [c for c in df_full.columns if 'score_' in c.lower() or 'балл' in c.lower()]
        if factor_cols:
            avg_scores = df_full[factor_cols].mean().reset_index()
            avg_scores.columns = ['Фактор', 'Средний балл']
            fig_scores = px.bar(avg_scores, x='Фактор', y='Средний балл', 
                                title="Средние баллы по факторам", height=300)
            st.plotly_chart(fig_scores, use_container_width=True)
        else:
            # Альтернатива: гистограмма вероятности
            fig_hist = px.histogram(df_full, x='final_probability', nbins=30, 
                                    title="Распределение вероятности ухода", height=300)
            st.plotly_chart(fig_hist, use_container_width=True)
    
    # Адаптивные заголовки KPI в зависимости от режима
    if "Активные" in scoring_mode:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Всего активных клиентов", value=data['summary']['total_clients'])
        with col2:
            st.metric(label="Высокий риск ухода", value=data['summary']['high_risk'], delta_color="inverse")
        with col3:
            st.metric(label="Средняя вероятность ухода", value=f"{data['summary']['avg_risk']:.2%}")
            
        risk_label = "Уровень риска"
        prob_col = "final_probability"
        top_factors_col = "top_factors"
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Всего отключившихся", value=data['summary']['total_clients'])
        with col2:
            st.metric(label="Высокая возвращаемость", value=data['summary']['high_risk'])
        with col3:
            st.metric(label="Средняя вероятность возврата", value=f"{data['summary']['avg_risk']:.2%}")
            
        risk_label = "Вероятность возврата"
        prob_col = "return_probability"
        top_factors_col = "return_reasons"

    st.divider()
    
    # 2. ФИЛЬТРЫ 
    st.subheader("Фильтры")
    col_f1, col_f2, col_f3, col_f4, col_f5 = st.columns(5)
    
    with col_f1:
        risk_levels = ["Высокий", "Средний", "Низкий"]
        selected_risks = st.multiselect("Уровень риска", options=risk_levels, default=risk_levels)
    
    with col_f2:
        filter_competitor = st.checkbox("Есть конкурент")
    
    with col_f3:
        filter_threat = st.checkbox("Была угроза")
    
    with col_f4:
        filter_low_activity = st.checkbox("Низкая активность")
    
    with col_f5:
        filter_incomplete = st.checkbox("Неполные данные")
    
    # Применение фильтров
    mask = df_full['risk_level'].isin(selected_risks)
    if filter_competitor:
        mask &= df_full.get('has_competitor', False) == True
    if filter_threat:
        mask &= df_full.get('has_threat', False) == True
    if filter_low_activity:
        # Условие низкой активности (например, время сессий < медианы или квантиля)
        median_session = df_full.get('total_session_minutes', pd.Series([0])).median()
        mask &= df_full.get('total_session_minutes', 0) <= median_session
    if filter_incomplete:
        # Условие неполных данных (например, feature_completeness < 0.8)
        mask &= df_full.get('feature_completeness', 1) < 0.8
        
    df_filtered = df_full[mask].copy()
    
    # 3. ТАБЛИЦА РЕЗУЛЬТАТОВ
        filter_threat = st.checkbox("Была угроза отключения")
        if filter_threat:
            mask &= df_full['has_threat'] == True

    df_filtered = df_full[mask].copy()
    if "final_probability" in df_filtered.columns:
        df_filtered["final_probability"] = (df_filtered["final_probability"] * 100).round(2).astype(str) + "%"

    if "feature_completeness" in df_filtered.columns:
        df_filtered["feature_completeness"] = (df_filtered["feature_completeness"] * 100).round(2).astype(str) + "%"
    # ТАБЛИЦА 
    st.subheader("Результаты скоринга")
    
    display_cols = [
        'company_name', 'code_to', 'final_probability', 'risk_level', 
        'feature_completeness', 'top_factors', 'competitor', 'has_threat',
        'total_session_minutes'
    ]
    safe_cols = [c for c in display_cols if c in df_filtered.columns]
    
    st.dataframe(df_filtered[safe_cols], use_container_width=True, hide_index=True)
    
    # 4. ЭКСПОРТ В EXCEL
    rename_columns = {
    "company_name": "Контрагент",
    "code_to": "Код ТО",
    "final_probability": "Вероятность ухода",
    "return_probability": "Вероятность возврата",
    "risk_level": "Уровень риска",
    "return_level": "Уровень возвращаемости",
    "feature_completeness": "Полнота данных",
    "top_factors": "Основные факторы",
    "return_reasons": "Причины возврата",
    "competitor": "Конкурент",
    "has_threat": "Была угроза отключения"
    }

    df_show = df_filtered[safe_cols].rename(columns=rename_columns)

    st.dataframe(df_show, use_container_width=True, hide_index=True)
   

    # ЭКСПОРТ В EXCEL
    if not df_filtered.empty:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            sheet_name = "ML_Scoring" if current_mode == 'ml' else "Manual_Scoring"
            df_filtered.to_excel(writer, index=False, sheet_name=sheet_name)
            
        excel_bytes = output.getvalue()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        download_filename = f"scoring_{current_mode}_{timestamp}.xlsx"
        
        st.download_button(
            label="Скачать отчет в Excel",
            data=excel_bytes,
            file_name=download_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_excel_btn"
        )

else:
    # ИНТЕРФЕЙС НАСТРОЙКИ РУЧНОГО СКОРИНГА (ПОКАЗЫВАЕТСЯ ТОЛЬКО ЕСЛИ НЕТ РЕЗУЛЬТАТОВ И ВЫБРАН РУЧНОЙ РЕЖИМ)
    if st.session_state['scoring_mode'] == 'manual':
        st.subheader("️ Конструктор ручного скоринга")
        st.info("Настройте признаки и веса ПЕРЕД загрузкой файла. Сумма весов должна быть равна 1.0")
        
        features = st.session_state['manual_features']
        
        # ОТОБРАЖЕНИЕ ТЕКУЩИХ ПРИЗНАКОВ
        for i, feat in enumerate(features):
            cols = st.columns([4, 2, 2, 2, 1])
            
            with cols[0]:
                st.text(feat['name'])
            
            with cols[1]:
                new_weight = st.number_input(
                    "Вес", 
                    value=feat['weight'], 
                    min_value=0.0, max_value=1.0, step=0.01,
                    key=f"weight_{i}",
                    label_visibility="collapsed"
                )
                feat['weight'] = round(new_weight, 4)
                
            with cols[2]:
                feat['type'] = st.selectbox(
                    "Тип", 
                    options=FEATURE_TYPES,
                    index=FEATURE_TYPES.index(feat['type']),
                    key=f"type_{i}",
                    label_visibility="collapsed"
                )
                
            with cols[3]:
                feat['formula'] = st.selectbox(
                    "Формула", 
                    options=list(FORMULA_OPTIONS.keys()),
                    format_func=lambda x: FORMULA_OPTIONS[x],
                    index=list(FORMULA_OPTIONS.keys()).index(feat['formula']),
                    key=f"formula_{i}",
                    label_visibility="collapsed"
                )
                
            with cols[4]:
                if st.button("🗑️", key=f"del_{i}"):
                    features.pop(i)
                    st.rerun()
        
        # КОНТРОЛЬ СУММЫ ВЕСОВ
        total_weight = sum(f['weight'] for f in features)
        st.divider()
        weight_status = "" if abs(total_weight - 1.0) < 0.001 else "❌"
        st.metric("Сумма весов", f"{total_weight:.4f} / 1.0000", delta=f"{weight_status} {'OK' if abs(total_weight - 1.0) < 0.001 else 'Требуется корректировка'}")
        
        # АВТОКОРРЕКЦИЯ ВЕСОВ
        if abs(total_weight - 1.0) > 0.001 and total_weight > 0:
            if st.button("Автокорректировать веса до 1.0"):
                scale_factor = 1.0 / total_weight
                for f in features:
                    f['weight'] = round(f['weight'] * scale_factor, 4)
                # Корректируем последний признак, чтобы убрать ошибку округления
                features[-1]['weight'] = round(1.0 - sum(f['weight'] for f in features[:-1]), 4)
                st.rerun()
        
        # ДОБАВЛЕНИЕ НОВОГО ПРИЗНАКА
        st.divider()
        st.subheader("Добавить новый признак")
        add_cols = st.columns([4, 2, 2, 2, 1])
        
        with add_cols[0]:
            new_name = st.text_input("Название признака", key="new_feat_name")
        with add_cols[1]:
            new_weight_val = st.number_input("Вес", min_value=0.01, max_value=1.0, value=0.1, step=0.01, key="new_feat_weight")
        with add_cols[2]:
            new_type = st.selectbox("Тип", FEATURE_TYPES, key="new_feat_type")
        with add_cols[3]:
            new_formula = st.selectbox("Формула", list(FORMULA_OPTIONS.keys()), format_func=lambda x: FORMULA_OPTIONS[x], key="new_feat_formula")
        with add_cols[4]:
            if st.button("Добавить", type="primary", key="add_feat_btn"):
                if new_name:
                    features.append({
                        "name": new_name,
                        "weight": new_weight_val,
                        "type": new_type,
                        "formula": new_formula
                    })
                    st.rerun()
                else:
                    st.warning("Введите название признака")
    
    else:
        # ML РЕЖИМ БЕЗ РЕЗУЛЬТАТОВ
        st.info("Загрузите Excel файл в боковой панели и нажмите «Рассчитать ML скоринг», чтобы увидеть результаты.")