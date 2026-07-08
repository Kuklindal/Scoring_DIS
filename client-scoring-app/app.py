import streamlit as st
import pandas as pd
import requests
import io
from datetime import datetime


# КОНФИГУРАЦИЯ API
# Заменить на реальный адрес бэкенда после запуска!!!
#API_BASE_URL = "http://localhost:8000" 
st.set_page_config(page_title="Churn & Win-back System", layout="wide")
st.title("Система скоринга клиентов ДИС")

# ВЫБОР РЕЖИМА СКОРИНГА
scoring_mode = st.radio(
    "Выберите тип анализа:",
    options=["Скоринг риска оттока (Активные)", "Скоринг возвращаемости (Отключившиеся)"],
    index=0,
    horizontal=True,
    label_visibility="collapsed"
)

# Инициализация состояния
if 'result_data' not in st.session_state:
    st.session_state['result_data'] = None
if 'current_mode' not in st.session_state:
    st.session_state['current_mode'] = scoring_mode

# Если режим изменился, сбрасываем старые результаты
if st.session_state['current_mode'] != scoring_mode:
    st.session_state['result_data'] = None
    st.session_state['current_mode'] = scoring_mode

# БОКОВАЯ ПАНЕЛЬ 
with st.sidebar:
    st.header(f"Загрузка данных для: {scoring_mode}")
    
    # Разные подсказки в зависимости от режима
    if "Активные" in scoring_mode:
        help_text = "Загрузите файл КИС_дляСкорингаДИС_Сопр.xlsx"
    else:
        help_text = "Загрузите файл КИС_дляСкорингаДИС_Откл.xlsx"
        
    uploaded_file = st.file_uploader("Excel файл (.xlsx)", type=["xlsx"], help=help_text)
    
    if uploaded_file is not None:
        st.success(f"Файл выбран")
        
        if st.button("Рассчитать скоринг", type="primary"):
            with st.spinner("Обработка данных..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
                    
                    # Передаем параметр mode на бэкенд
                    params = {"mode": "churn" if "Активные" in scoring_mode else "winback"}
                    
                    response = requests.post(f"{API_BASE_URL}/score", files=files, params=params)
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state['result_data'] = data
                        st.success("Расчет завершен!")
                    else:
                        st.error(f"Ошибка ({response.status_code}): {response.text}")
                        
                except Exception as e:
                    st.error(f"Ошибка соединения: {str(e)}")

# --- ОСНОВНОЙ ЭКРАН ---
if st.session_state['result_data']:
    data = st.session_state['result_data']
    
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
            st.metric(label="Высокая возвращаемость", value=data['summary']['high_return'])
        with col3:
            st.metric(label="Средняя вероятность возврата", value=f"{data['summary']['avg_return']:.2%}")
            
        risk_label = "Вероятность возврата"
        prob_col = "return_probability"
        top_factors_col = "return_reasons"

    st.divider()

    # ФИЛЬТРЫ (АДАПТИВНЫЕ)
    st.subheader("Фильтры")
    clients_list = data.get('clients', [])
    df_full = pd.DataFrame(clients_list)
    
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        if "Активные" in scoring_mode:
            levels = ["Высокий", "Средний", "Низкий"]
            selected_levels = st.multiselect(risk_label, options=levels, default=levels)
            mask = df_full['risk_level'].isin(selected_levels)
        else:
            levels = ["Высокая", "Средняя", "Низкая"]
            selected_levels = st.multiselect(risk_label, options=levels, default=levels)
            mask = df_full['return_level'].isin(selected_levels)

    with col_f2:
        filter_competitor = st.checkbox("Есть конкурент")
        if filter_competitor:
            mask &= df_full['has_competitor'] == True
            
    with col_f3:
        filter_threat = st.checkbox("Была угроза")
        if filter_threat:
            mask &= df_full['has_threat'] == True

    df_filtered = df_full[mask].copy()

    # ТАБЛИЦА 
    st.subheader("Результаты скоринга")
    
    display_cols = [
        'company_name', 'code_to', prob_col, risk_label.replace("Вероятность ", "").replace("Уровень ", ""), 
        'feature_completeness', top_factors_col, 'competitor', 'has_threat'
    ]
    safe_cols = [c for c in display_cols if c in df_filtered.columns]
    
    st.dataframe(df_filtered[safe_cols], use_container_width=True, hide_index=True)

    # ЭКСПОРТ В EXCEL
    if not df_filtered.empty:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            sheet_name = "Churn_Risk" if "Активные" in scoring_mode else "Winback_Score"
            df_filtered.to_excel(writer, index=False, sheet_name=sheet_name)
            
        excel_bytes = output.getvalue()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        download_filename = f"scoring_{scoring_mode[:6]}_{timestamp}.xlsx"
        
        st.download_button(
            label="Скачать отчет в Excel",
            data=excel_bytes,
            file_name=download_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_excel_btn"
        )
else:
    st.info(f"Загрузите файл для режима **«{scoring_mode}»** и нажмите «Рассчитать скоринг».")