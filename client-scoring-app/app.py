import io
import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_BASE_URL = "http://localhost:8000"

# --- ДЕФОЛТНЫЕ ПРИЗНАКИ ДЛЯ АКТИВНЫХ КЛИЕНТОВ (СОПР) ---
DEFAULT_FEATURES_ACTIVE = [
    {
        "name": "Конкурент",
        "group": "risk",
        "risk_weight": 0.30,
        "value_weight": 0.00,
        "type": "Бинарный",
        "formula": "binary",
    },
    {
        "name": "Была угроза отключения",
        "group": "risk",
        "risk_weight": 0.30,
        "value_weight": 0.00,
        "type": "Бинарный",
        "formula": "binary",
    },
    {
        "name": "Время всех сессий в мин",
        "group": "both",
        "risk_weight": 0.20,
        "value_weight": 0.20,
        "type": "Числовой",
        "formula": "min_max_inverse",
    },
    {
        "name": "Количество ОС",
        "group": "both",
        "risk_weight": 0.05,
        "value_weight": 0.10,
        "type": "Числовой",
        "formula": "step_os",
    },
    {
        "name": "Срок жизни от регистрации",
        "group": "both",
        "risk_weight": 0.10,
        "value_weight": 0.10,
        "type": "Числовой",
        "formula": "min_max",
    },
    {
        "name": "%Скидки",
        "group": "both",
        "risk_weight": 0.05,
        "value_weight": 0.10,
        "type": "Процентный",
        "formula": "min_max_inverse",
    },
    {
        "name": "Численность сотрудников",
        "group": "value",
        "risk_weight": 0.00,
        "value_weight": 0.025,
        "type": "Числовой",
        "formula": "min_max",
    },
    {
        "name": "Сумма счета ИО",
        "group": "value",
        "risk_weight": 0.00,
        "value_weight": 0.25,
        "type": "Числовой",
        "formula": "min_max",
    },
    {
        "name": "Годовая выручка",
        "group": "value",
        "risk_weight": 0.00,
        "value_weight": 0.225,
        "type": "Числовой",
        "formula": "min_max",
    },
]

# --- ДЕФОЛТНЫЕ ПРИЗНАКИ ДЛЯ УШЕДШИХ КЛИЕНТОВ (ОТКЛ / WIN-BACK) ---
DEFAULT_FEATURES_CHURNED = [
    {
        "name": "Причина отключения",
        "group": "return",
        "risk_weight": 0.0,
        "value_weight": 0.15,
        "type": "Категориальный",
        "formula": "reason_multiplier",
    },
    {
        "name": "Сумма счета ИО",
        "group": "return",
        "risk_weight": 0.0,
        "value_weight": 0.15,
        "type": "Числовой",
        "formula": "min_max",
    },
    {
        "name": "Время всех сессий в мин",
        "group": "return",
        "risk_weight": 0.0,
        "value_weight": 0.15,
        "type": "Числовой",
        "formula": "min_max",
    },
    {
        "name": "Конкурент",
        "group": "return",
        "risk_weight": 0.0,
        "value_weight": 0.10,
        "type": "Бинарный",
        "formula": "binary_inverse",
    },
    {
        "name": "Количество ОС",
        "group": "return",
        "risk_weight": 0.0,
        "value_weight": 0.10,
        "type": "Числовой",
        "formula": "min_max",
    },
    {
        "name": "Годовая выручка",
        "group": "return",
        "risk_weight": 0.0,
        "value_weight": 0.10,
        "type": "Числовой",
        "formula": "min_max",
    },
    {
        "name": "Срок жизни от регистрации",
        "group": "return",
        "risk_weight": 0.0,
        "value_weight": 0.075,
        "type": "Числовой",
        "formula": "min_max",
    },
    {
        "name": "%Скидки",
        "group": "return",
        "risk_weight": 0.0,
        "value_weight": 0.075,
        "type": "Процентный",
        "formula": "min_max_inverse",
    },
    {
        "name": "Численность сотрудников",
        "group": "return",
        "risk_weight": 0.0,
        "value_weight": 0.05,
        "type": "Числовой",
        "formula": "min_max",
    },
    {
        "name": "Вид деятельности",
        "group": "return",
        "risk_weight": 0.0,
        "value_weight": 0.05,
        "type": "Категориальный",
        "formula": "categorical",
    },
    {
        "name": "Была угроза отключения",
        "group": "return",
        "risk_weight": 0.0,
        "value_weight": 0.05,
        "type": "Бинарный",
        "formula": "binary_inverse",
    },
]

FORMULA_OPTIONS = {
    "binary": "Бинарный (0/1)",
    "binary_inverse": "Бинарный инвертированный (пусто=1)",
    "min_max": "Min-Max нормализация",
    "min_max_inverse": "Min-Max инвертированная",
    "step_os": "Ступенчатая (ОС: 1=30, 2=70, 3+=100)",
    "reason_multiplier": "Множитель причины отключения",
    "categorical": "Категориальный",
}

FEATURE_TYPES = ["Бинарный", "Числовой", "Процентный", "Категориальный"]
FEATURE_GROUPS = {
    "risk": "Риск",
    "value": "Ценность",
    "both": "Риск и ценность",
    "return": "Возвращаемость",
}
RISK_LEVELS_ALL = ["Низкий", "Средний", "Высокий", "Критический"]
RETURN_LEVELS_ALL = ["Низкая", "Средняя", "Высокая", "Очень высокая"]

st.set_page_config(
    page_title="Scoring DIS System",
    layout="wide",
    page_icon="📊",
)

st.title("Система скоринга клиентов ДИС")


def init_session_state() -> None:
    if "scoring_mode" not in st.session_state:
        st.session_state["scoring_mode"] = "ml"

    # Добавляем состояние типа клиента (active / churned)
    if "client_type" not in st.session_state:
        st.session_state["client_type"] = "active"

    # Выбираем дефолтные признаки в зависимости от типа клиента
    default_feats = DEFAULT_FEATURES_ACTIVE if st.session_state["client_type"] == "active" else DEFAULT_FEATURES_CHURNED

    if "manual_features" not in st.session_state:
        st.session_state["manual_features"] = [
            feature.copy() for feature in default_feats
        ]

    if "result_data" not in st.session_state:
        st.session_state["result_data"] = None

    if "uploader_version" not in st.session_state:
        st.session_state["uploader_version"] = 0


def sync_manual_features_from_widgets() -> None:
    features = st.session_state.get("manual_features", [])

    for index, feature in enumerate(features):
        mapping = {
            "group": f"group_{index}",
            "risk_weight": f"risk_weight_{index}",
            "value_weight": f"value_weight_{index}",
            "type": f"type_{index}",
            "formula": f"formula_{index}",
        }

        for field, key in mapping.items():
            if key in st.session_state:
                feature[field] = st.session_state[key]


def get_group_total(group: str) -> float:
    """
    Суммирует реальные веса независимо от выбранной группы.
    risk_weight > 0 означает участие признака в скоре риска.
    value_weight > 0 означает участие признака в скоре ценности.
    """
    weight_field = (
        "risk_weight"
        if group == "risk"
        else "value_weight"
    )

    return round(
        sum(
            float(feature.get(weight_field, 0) or 0)
            for feature in st.session_state.get(
                "manual_features",
                [],
            )
        ),
        3,
    )


def manual_weights_are_valid() -> bool:
    features = st.session_state.get("manual_features", [])
    # Для ушедших клиентов проверяем только вес возвращаемости (value_weight)
    target_group = "risk" if st.session_state["client_type"] == "active" else "value"
    total = get_group_total(target_group)

    if st.session_state["client_type"] == "active":
        return (
                bool(features)
                and abs(get_group_total("risk") - 1.0) < 0.005
                and abs(get_group_total("value") - 1.0) < 0.005
        )
    else:
        return bool(features) and abs(total - 1.0) < 0.005


def send_scoring_request(
        uploaded_file,
        current_mode: str,
        client_type: str,
) -> dict | None:
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    }

    if current_mode == "ml":
        # Адаптация эндпоинта под тип клиента
        endpoint_suffix = "winback" if client_type == "churned" else "churn"
        endpoint = f"{API_BASE_URL}/score/{endpoint_suffix}"
        params = None
    else:
        target_group = "risk" if client_type == "active" else "value"
        total = get_group_total(target_group)

        if abs(total - 1.0) >= 0.005:
            label = "Риска" if client_type == "active" else "Возвращаемости"
            st.error(
                f"Сумма весов ({label}) должна быть равна 1.00. "
                f"Текущая сумма: {total:.3f}"
            )
            return None

        config = {
            "features": st.session_state["manual_features"],
            "target_column": "final_probability",
        }

        files["config"] = (
            None,
            json.dumps(config, ensure_ascii=False),
            "application/json",
        )
        endpoint = f"{API_BASE_URL}/score/manual"
        params = None

    try:
        response = requests.post(
            endpoint,
            files=files,
            params=params,
            timeout=120,
        )
    except requests.RequestException as exc:
        st.error(f"Ошибка соединения с backend: {exc}")
        return None

    if response.status_code != 200:
        st.error(
            f"Ошибка сервера ({response.status_code}): {response.text}"
        )
        return None

    try:
        return response.json()
    except ValueError:
        st.error("Backend вернул ответ, который не является JSON.")
        return None


def clear_results_callback() -> None:
    """
    Полный сброс интерфейса:
    - очищает результат расчёта;
    - возвращает дефолтные признаки для ТЕКУЩЕГО типа клиента;
    - удаляет состояния связанных виджетов;
    - пересоздаёт file_uploader, чтобы выбранный файл исчез.
    """
    st.session_state["result_data"] = None

    default_feats = DEFAULT_FEATURES_ACTIVE if st.session_state["client_type"] == "active" else DEFAULT_FEATURES_CHURNED
    st.session_state["manual_features"] = [
        feature.copy() for feature in default_feats
    ]

    keys_to_delete = []

    for key in list(st.session_state.keys()):
        if (
                key.startswith("risk_weight_")
                or key.startswith("value_weight_")
                or key.startswith("group_")
                or key.startswith("type_")
                or key.startswith("formula_")
                or key.startswith("delete_feature_")
                or key in {
            "new_feature_name",
            "new_feature_group",
            "new_feature_risk_weight",
            "new_feature_value_weight",
            "new_feature_type",
            "new_feature_formula",
            "normalize_manual_weights",
        }
        ):
            keys_to_delete.append(key)

    for key in keys_to_delete:
        del st.session_state[key]

    st.session_state["uploader_version"] += 1


def normalize_manual_weights_callback() -> None:
    features = st.session_state.get("manual_features", [])
    target_group = "risk" if st.session_state["client_type"] == "active" else "value"
    weight_field = "risk_weight" if target_group == "risk" else "value_weight"

    indices = []
    for index, feature in enumerate(features):
        widget_value = float(
            st.session_state.get(
                f"{weight_field}_{index}",
                feature.get(weight_field, 0.0),
            )
            or 0.0
        )
        if widget_value > 0:
            indices.append(index)

    if not indices:
        return

    current_weights = [
        float(
            st.session_state.get(
                f"{weight_field}_{index}",
                features[index].get(weight_field, 0.0),
            )
        )
        for index in indices
    ]

    total = sum(current_weights)
    if total <= 0:
        return

    corrected = [round(weight / total, 3) for weight in current_weights]
    corrected[-1] = round(1.0 - sum(corrected[:-1]), 3)

    for index, corrected_weight in zip(indices, corrected):
        features[index][weight_field] = corrected_weight
        st.session_state[f"{weight_field}_{index}"] = corrected_weight

    st.session_state["manual_features"] = features


def render_manual_scoring_constructor() -> None:
    st.subheader("Конструктор ручного скоринга")

    client_label = "активных клиентов (Риск + Ценность)" if st.session_state[
                                                                "client_type"] == "active" else "ушедших клиентов (Возвращаемость)"
    st.info(
        f"Настройка формулы для {client_label}. "
        "Сумма весов должна быть равна 1,00."
    )

    features = st.session_state["manual_features"]

    if not features:
        st.warning("Добавьте хотя бы один признак.")

    header = st.columns([3.0, 1.6, 1.25, 1.25, 1.5, 2.3, 0.55])
    labels = [
        "Признак",
        "Группа",
        "Вес риска",
        "Вес ценности",
        "Тип",
        "Формула",
        "",
    ]
    for column, label in zip(header, labels):
        column.caption(label)

    formula_keys = list(FORMULA_OPTIONS.keys())

    for index, feature in enumerate(features):
        columns = st.columns([3.0, 1.6, 1.25, 1.25, 1.5, 2.3, 0.55])

        with columns[0]:
            st.text(feature["name"])

        with columns[1]:
            group = feature.get("group", "risk")
            feature["group"] = st.selectbox(
                "Группа",
                options=list(FEATURE_GROUPS.keys()),
                format_func=lambda value: FEATURE_GROUPS[value],
                index=list(FEATURE_GROUPS.keys()).index(group),
                key=f"group_{index}",
                label_visibility="collapsed",
            )

        with columns[2]:
            feature["risk_weight"] = st.number_input(
                "Вес риска",
                min_value=0.0,
                max_value=1.0,
                value=float(feature.get("risk_weight", 0.0)),
                step=0.005,
                format="%.3f",
                key=f"risk_weight_{index}",
                label_visibility="collapsed",
            )

        with columns[3]:
            feature["value_weight"] = st.number_input(
                "Вес ценности",
                min_value=0.0,
                max_value=1.0,
                value=float(feature.get("value_weight", 0.0)),
                step=0.005,
                format="%.3f",
                key=f"value_weight_{index}",
                label_visibility="collapsed",
            )

        with columns[4]:
            current_type = feature.get("type", FEATURE_TYPES[0])
            feature["type"] = st.selectbox(
                "Тип",
                options=FEATURE_TYPES,
                index=FEATURE_TYPES.index(current_type),
                key=f"type_{index}",
                label_visibility="collapsed",
            )

        with columns[5]:
            current_formula = feature.get(
                "formula",
                "binary" if feature["type"] == "Бинарный" else "min_max",
            )
            if current_formula not in formula_keys:
                current_formula = formula_keys[0]

            feature["formula"] = st.selectbox(
                "Формула",
                options=formula_keys,
                format_func=lambda value: FORMULA_OPTIONS[value],
                index=formula_keys.index(current_formula),
                key=f"formula_{index}",
                label_visibility="collapsed",
            )

        with columns[6]:
            if st.button("🗑️", key=f"delete_feature_{index}"):
                features.pop(index)
                st.rerun()

    target_group = "risk" if st.session_state["client_type"] == "active" else "value"
    total = get_group_total(target_group)

    st.divider()

    metric_columns = st.columns(2)
    label_text = "Риска" if target_group == "risk" else "Возвращаемости"

    with metric_columns[0]:
        st.metric(
            f"Сумма весов ({label_text})",
            f"{total:.3f} / 1.000",
            delta=(
                "OK"
                if abs(total - 1.0) < 0.005
                else "Требуется корректировка"
            ),
        )

    if abs(total - 1.0) >= 0.005:
        st.button(
            f"Автокорректировать веса ({label_text})",
            key="normalize_manual_weights",
            on_click=normalize_manual_weights_callback,
            use_container_width=True,
        )

    st.divider()
    st.subheader("Добавить новый признак")

    add_columns = st.columns([2.8, 1.5, 1.2, 1.2, 1.4, 2.2])

    with add_columns[0]:
        new_name = st.text_input(
            "Название признака",
            key="new_feature_name",
        )

    with add_columns[1]:
        new_group = st.selectbox(
            "Группа",
            options=list(FEATURE_GROUPS.keys()),
            format_func=lambda value: FEATURE_GROUPS[value],
            key="new_feature_group",
        )

    with add_columns[2]:
        new_risk_weight = st.number_input(
            "Вес риска",
            min_value=0.0,
            max_value=1.0,
            value=0.10,
            step=0.005,
            format="%.3f",
            key="new_feature_risk_weight",
        )

    with add_columns[3]:
        new_value_weight = st.number_input(
            "Вес ценности",
            min_value=0.0,
            max_value=1.0,
            value=0.10,
            step=0.005,
            format="%.3f",
            key="new_feature_value_weight",
        )

    with add_columns[4]:
        new_type = st.selectbox(
            "Тип",
            FEATURE_TYPES,
            key="new_feature_type",
        )

    with add_columns[5]:
        new_formula = st.selectbox(
            "Формула",
            formula_keys,
            format_func=lambda value: FORMULA_OPTIONS[value],
            key="new_feature_formula",
        )

    if st.button(
            "Добавить признак",
            type="primary",
            use_container_width=True,
    ):
        if not new_name.strip():
            st.warning("Введите название признака.")
        elif any(
                feature["name"].strip().lower()
                == new_name.strip().lower()
                for feature in features
        ):
            st.warning("Признак с таким названием уже существует.")
        else:
            features.append(
                {
                    "name": new_name.strip(),
                    "group": new_group,
                    "risk_weight": round(float(new_risk_weight), 3),
                    "value_weight": round(float(new_value_weight), 3),
                    "type": new_type,
                    "formula": new_formula,
                }
            )
            st.rerun()


def render_dashboard(data: dict, current_mode: str, client_type: str) -> None:
    clients = data.get("clients", [])
    summary = data.get("summary", {})
    df_full = pd.DataFrame(clients)

    if df_full.empty:
        st.warning("Backend не вернул клиентов для отображения.")
        return

    prob_col = "final_probability"
    is_churned = client_type == "churned"

    # Адаптация названий колонок и метрик под тип клиента
    risk_col = "risk_level" if not is_churned else "return_level"

    if is_churned:
        total_label = "Всего ушедших"
        high_label = "Высокая возвращаемость"
        average_label = "Средняя возвращаемость"
        probability_label = "Вероятность возврата"
        levels = RETURN_LEVELS_ALL
    else:
        total_label = "Всего активных"
        high_label = "Высокий риск ухода"
        average_label = "Средний риск ухода"
        probability_label = "Вероятность ухода"
        levels = RISK_LEVELS_ALL

    st.subheader("Дашборд скоринга")

    columns = st.columns(4)

    total_clients = summary.get("total_clients", len(df_full))

    if risk_col in df_full.columns:
        high_targets = ["Высокий", "Критический"] if not is_churned else ["Высокая", "Очень высокая"]
        high_risk_default = int(
            df_full[risk_col].isin(high_targets).sum()
        )
    else:
        high_risk_default = 0

    high_risk = summary.get("high_risk", high_risk_default)

    if prob_col in df_full.columns:
        probability_series = pd.to_numeric(
            df_full[prob_col],
            errors="coerce",
        )
        avg_probability_default = float(probability_series.mean())
        if pd.isna(avg_probability_default):
            avg_probability_default = 0.0
    else:
        avg_probability_default = 0.0

    avg_probability = float(
        summary.get("avg_risk", avg_probability_default)
    )

    if "feature_completeness" in df_full.columns:
        completeness = pd.to_numeric(
            df_full["feature_completeness"],
            errors="coerce",
        ).mean()
        avg_completeness = 0.0 if pd.isna(completeness) else float(completeness)
    else:
        avg_completeness = 0.0

    with columns[0]:
        st.metric(total_label, total_clients)

    with columns[1]:
        st.metric(high_label, high_risk, delta_color="inverse")

    with columns[2]:
        st.metric(average_label, f"{avg_probability:.2%}")

    with columns[3]:
        st.metric("Средняя полнота данных", f"{avg_completeness:.2%}")

    chart_columns = st.columns([1, 1.55])

    risk_color_map = {
        "Низкий": "#2ecc71",
        "Средний": "#f1c40f",
        "Высокий": "#e67e22",
        "Критический": "#e74c3c",
        "Не определён": "#95a5a6",
        "Низкая": "#2ecc71",
        "Средняя": "#f1c40f",
        "Высокая": "#e67e22",
        "Очень высокая": "#9b59b6",
    }

    with chart_columns[0]:
        if risk_col in df_full.columns:
            risk_counts = (
                df_full[risk_col]
                .fillna("Не определён")
                .value_counts()
                .reindex(levels + ["Не определён"], fill_value=0)
                .reset_index()
            )
            risk_counts.columns = ["Уровень", "Количество"]
            risk_counts = risk_counts[risk_counts["Количество"] > 0]

            title_text = "Распределение по уровням риска" if not is_churned else "Распределение по уровням возвращаемости"

            figure = px.pie(
                risk_counts,
                values="Количество",
                names="Уровень",
                title=title_text,
                color="Уровень",
                color_discrete_map=risk_color_map,
                category_orders={"Уровень": levels + ["Не определён"]},
            )
            figure.update_traces(
                textinfo="percent",
                textposition="inside",
                textfont=dict(size=18),
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Клиентов: %{value}<br>"
                    "Доля: %{percent}<extra></extra>"
                ),
                marker=dict(line=dict(width=1)),
            )
            figure.update_layout(
                height=500,
                legend_title_text="Уровень",
                legend=dict(
                    orientation="v",
                    x=1.02,
                    y=0.5,
                    yanchor="middle",
                    font=dict(size=16),
                    title_font=dict(size=17),
                    itemsizing="constant",
                ),
                margin=dict(l=20, r=165, t=70, b=20),
            )
            st.plotly_chart(figure, use_container_width=True)
        else:
            st.info(f"В ответе backend нет поля {risk_col}.")

    with chart_columns[1]:
        excluded_chart_columns = {
            "client_id",
            "company_name",
            "code_to",
            prob_col,
            risk_col,
            "feature_completeness",
            "top_factors",
            "disconnection_reason",
        }

        chart_labels = {
            "life_span": "Срок жизни от регистрации",
            "number_os": "Количество ОС",
            "summa_scheta_io": "Сумма счёта ИО",
            "discount": "% скидки",
            "%Скидки": "% скидки",
            "annual_revenue": "Годовая выручка",
            "number_of_employees": "Численность сотрудников",
            "total_session_minutes": "Время всех сессий, мин",
            "has_competitor": "Наличие конкурента",
            "has_threat": "Наличие угрозы отключения",
            "competitor": "Конкурент",
        }

        chart_option_map = {}

        if current_mode == "manual":
            selected_features = [
                str(feature_name).strip()
                for feature_name in data.get("selected_features", [])
            ]
            manual_feature_aliases = {
                "Конкурент": "has_competitor",
                "Была угроза отключения": "has_threat",
            }
            preferred_items = []
            for feature_name in selected_features:
                source_column = feature_name
                numeric_values = (
                    pd.to_numeric(df_full[source_column], errors="coerce")
                    if source_column in df_full.columns
                    else pd.Series(dtype=float)
                )
                if source_column not in df_full.columns or numeric_values.notna().sum() == 0:
                    alias_column = manual_feature_aliases.get(feature_name)
                    if alias_column and alias_column in df_full.columns:
                        source_column = alias_column
                    else:
                        continue
                preferred_items.append((source_column, feature_name))
        else:
            preferred_items = [
                (column, chart_labels.get(column, str(column)))
                for column in sorted(
                    df_full.columns,
                    key=lambda column: (0 if column == "%Скидки" else 1, str(column)),
                )
            ]

        for column, display_label in preferred_items:
            if column in excluded_chart_columns:
                continue
            numeric_values = pd.to_numeric(df_full[column], errors="coerce")
            if numeric_values.notna().sum() == 0:
                continue

            display_label = chart_labels.get(column, display_label)
            if current_mode == "manual":
                display_label = next(
                    (orig for src, orig in preferred_items if src == column),
                    display_label,
                )

            normalized_label = str(display_label).strip().lower().replace("ё", "е").replace(" ", "")
            if normalized_label not in chart_option_map:
                chart_option_map[normalized_label] = {"column": column, "label": display_label}

        graph_options = [f"Распределение {probability_label}"]
        graph_options.extend(item["column"] for item in chart_option_map.values())
        option_labels = {item["column"]: item["label"] for item in chart_option_map.values()}

        selected_graph = st.selectbox(
            f"Показать {probability_label} относительно:",
            options=graph_options,
            format_func=lambda value: (
                value if value == f"Распределение {probability_label}" else option_labels.get(value, value)
            ),
            key="score_distribution_feature",
        )

        graph_df = df_full.copy()
        graph_df[prob_col] = pd.to_numeric(graph_df[prob_col], errors="coerce")

        if selected_graph == f"Распределение {probability_label}":
            probability_values = graph_df[prob_col].dropna().clip(0, 1)
            counts, bin_edges = np.histogram(probability_values, bins=20, range=(0, 1))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            distribution_df = pd.DataFrame(
                {
                    "Диапазон": [f"{l:.0%}–{r:.0%}" for l, r in zip(bin_edges[:-1], bin_edges[1:])],
                    probability_label: bin_centers,
                    "Количество клиентов": counts,
                }
            )

            figure = px.bar(
                distribution_df,
                x="Диапазон",
                y="Количество клиентов",
                color=probability_label,
                color_continuous_scale=["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"],
                title=f"Распределение {probability_label}",
                hover_data={probability_label: ":.1%", "Количество клиентов": True},
            )
            figure.update_layout(height=500, coloraxis_showscale=False)
        else:
            feature_label = option_labels.get(selected_graph, chart_labels.get(selected_graph, selected_graph))
            graph_df[selected_graph] = pd.to_numeric(graph_df[selected_graph], errors="coerce")
            is_percent_feature = "%" in str(feature_label) or selected_graph in {"discount", "%Скидки"}
            graph_df = graph_df.dropna(subset=[selected_graph, prob_col])
            unique_count = graph_df[selected_graph].nunique()

            if graph_df.empty:
                figure = px.bar(pd.DataFrame({"Группа": [], "Средний итоговый скор": []}), x="Группа",
                                y="Средний итоговый скор", title=f"Нет данных для признака: {feature_label}")
            elif unique_count <= 10:
                graph_df["_Группа"] = graph_df[selected_graph].map(
                    lambda value: (
                        f"{value:.0%}" if is_percent_feature
                        else "Да" if value == 1
                        else "Нет" if value == 0
                        else f"{value:g}"
                    )
                )
                grouped_df = graph_df.groupby("_Группа", as_index=False).agg(
                    **{"Средний итоговый скор": (prob_col, "mean"), "Количество клиентов": (prob_col, "size")})
                figure = px.bar(grouped_df, x="_Группа", y="Средний итоговый скор", color="Средний итоговый скор",
                                color_continuous_scale=["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"],
                                title=f"Средний {probability_label} по признаку: {feature_label}",
                                labels={"_Группа": feature_label,
                                        "Средний итоговый скор": f"Средний {probability_label}"},
                                hover_data={"Количество клиентов": True, "Средний итоговый скор": ":.2%"})
            else:
                number_of_bins = min(10, unique_count)
                try:
                    graph_df["_Диапазон"] = pd.qcut(graph_df[selected_graph], q=number_of_bins, duplicates="drop")
                except ValueError:
                    graph_df["_Диапазон"] = pd.cut(graph_df[selected_graph], bins=number_of_bins, duplicates="drop")

                grouped_df = graph_df.groupby("_Диапазон", observed=True, as_index=False).agg(
                    **{"Средний итоговый скор": (prob_col, "mean"), "Количество клиентов": (prob_col, "size"),
                       "Среднее значение признака": (selected_graph, "mean"),
                       "Минимум признака": (selected_graph, "min"),
                       "Максимум признака": (selected_graph, "max")}).sort_values("Среднее значение признака")
                figure = px.line(grouped_df, x="Среднее значение признака", y="Средний итоговый скор", markers=True,
                                 title=f"Зависимость {probability_label} от признака: {feature_label}",
                                 labels={"Среднее значение признака": feature_label,
                                         "Средний итоговый скор": f"Средний {probability_label}"},
                                 hover_data={"Количество клиентов": True, "Минимум признака": ":.2f",
                                             "Максимум признака": ":.2f", "Средний итоговый скор": ":.2%"})
                figure.update_traces(line=dict(width=3), marker=dict(size=10))

            figure.update_layout(height=500, coloraxis_showscale=False, yaxis_tickformat=".0%",
                                 yaxis_title=f"Средний {probability_label}", xaxis_title=feature_label,
                                 xaxis_tickangle=0)
            if is_percent_feature:
                figure.update_xaxes(tickformat=".0%", title_text=feature_label)

        st.plotly_chart(figure, use_container_width=True)

    st.divider()
    st.subheader("Фильтры")

    filter_columns = st.columns([1.3, 1, 1, 1, 1, 1.35])

    available_levels = [l for l in levels if
                        l in df_full.get(risk_col, []).unique()] if risk_col in df_full.columns else []

    with filter_columns[0]:
        level_label = "Уровень риска" if not is_churned else "Уровень возвращаемости"
        sel_levels = st.multiselect(level_label, options=available_levels, default=available_levels)

    with filter_columns[1]:
        filter_competitor = st.checkbox("Есть конкурент")

    with filter_columns[2]:
        filter_threat = st.checkbox("Была угроза")

    # Специфичный фильтр для ушедших клиентов
    with filter_columns[3]:
        if is_churned and "disconnection_reason" in df_full.columns:
            reasons = sorted(df_full["disconnection_reason"].dropna().unique().tolist())
            sel_reasons = st.multiselect("Причина отключения", options=reasons, default=reasons)
        else:
            filter_low_activity = st.checkbox("Низкая активность")

    with filter_columns[4]:
        filter_incomplete = st.checkbox("Неполные данные")

    with filter_columns[5]:
        sort_option = st.selectbox(
            "Сортировка",
            options=["По уровню риска", "По итоговому скору", "По контрагенту", "По коду ТО"],
            key="results_sort_option",
        )

    mask = pd.Series(True, index=df_full.index)

    if risk_col in df_full.columns and sel_levels:
        mask &= df_full[risk_col].isin(sel_levels)
    elif risk_col in df_full.columns and not sel_levels:
        mask &= False

    competitor_column = next((c for c in ["has_competitor", "competitor"] if c in df_full.columns), None)
    if filter_competitor:
        if competitor_column:
            mask &= df_full[competitor_column].isin([True, 1, "1", "Да", "да", "true", "True"])
        else:
            st.warning("В данных нет поля competitor или has_competitor.")

    if filter_threat:
        if "has_threat" in df_full.columns:
            mask &= df_full["has_threat"].isin([True, 1, "1", "Да", "да", "true", "True"])
        else:
            st.warning("В данных нет поля has_threat.")

    # Фильтрация по причине отключения для ушедших
    if is_churned and "disconnection_reason" in df_full.columns and 'sel_reasons' in locals():
        mask &= df_full["disconnection_reason"].isin(sel_reasons)
    elif not is_churned and "total_session_minutes" in df_full.columns and 'filter_low_activity' in locals() and filter_low_activity:
        sess = pd.to_numeric(df_full["total_session_minutes"], errors="coerce")
        mask &= sess <= sess.median()

    if filter_incomplete:
        if "feature_completeness" in df_full.columns:
            mask &= pd.to_numeric(df_full["feature_completeness"], errors="coerce") < 0.8
        else:
            st.warning("В данных нет поля feature_completeness.")

    df_filtered = df_full.loc[mask].copy()
    if prob_col in df_filtered.columns:
        df_filtered[prob_col] = pd.to_numeric(df_filtered[prob_col], errors="coerce")

    if risk_col in df_filtered.columns:
        df_filtered[risk_col] = pd.Categorical(df_filtered[risk_col], categories=levels, ordered=True)

    risk_order = {level: idx + 1 for idx, level in enumerate(reversed(levels))}

    if sort_option == "По уровню риска":
        risk_values = df_filtered[risk_col].astype(str).str.strip()
        df_filtered["_risk_rank"] = risk_values.map(risk_order).fillna(0).astype(int)
        df_filtered = df_filtered.sort_values(by=["_risk_rank", prob_col], ascending=[False, False], kind="mergesort",
                                              na_position="last").drop(columns=["_risk_rank"])
    elif sort_option == "По итоговому скору":
        df_filtered = df_filtered.sort_values(by=prob_col, ascending=False, na_position="last")
    elif sort_option == "По контрагенту":
        df_filtered = df_filtered.sort_values(by="company_name", ascending=True, na_position="last")
    elif sort_option == "По коду ТО":
        df_filtered["code_to"] = pd.to_numeric(df_filtered["code_to"], errors="coerce")
        df_filtered = df_filtered.sort_values(by="code_to", ascending=True, na_position="last")

    st.subheader("Результаты скоринга")
    st.caption(f"Найдено клиентов: {len(df_filtered)}")

    if current_mode == "manual":
        display_columns = ["company_name", "code_to", risk_col, "feature_completeness", "top_factors",
                           competitor_column, "has_threat", "value_score", "risk_score", prob_col]
    else:
        display_columns = ["company_name", "code_to", prob_col, risk_col, "feature_completeness", "top_factors",
                           competitor_column, "has_threat"]

    if is_churned and "disconnection_reason" in df_filtered.columns:
        display_columns.insert(4, "disconnection_reason")

    display_columns = [c for c in display_columns if c and c in df_filtered.columns]
    df_show = df_filtered[display_columns].copy()

    for score_column in ["value_score", "risk_score", prob_col]:
        if score_column in df_show.columns:
            df_show[score_column] = pd.to_numeric(df_show[score_column], errors="coerce")

    if "feature_completeness" in df_show.columns:
        df_show["feature_completeness"] = pd.to_numeric(df_show["feature_completeness"], errors="coerce")

    rename_columns = {
        "company_name": "Контрагент",
        "code_to": "Код ТО",
        "value_score": "Скор ценности",
        "risk_score": "Скор риска",
        prob_col: probability_label,
        risk_col: "Уровень",
        "feature_completeness": "Полнота данных",
        "top_factors": "Основные факторы",
        "competitor": "Конкурент",
        "has_competitor": "Конкурент",
        "has_threat": "Была угроза отключения",
        "disconnection_reason": "Причина отключения",
        "total_session_minutes": "Время сессий, мин",
    }
    df_show.rename(columns=rename_columns, inplace=True)

    dataframe_config = {
        "Скор ценности": st.column_config.NumberColumn("Скор ценности", format="percent"),
        "Скор риска": st.column_config.NumberColumn("Скор риска", format="percent"),
        probability_label: st.column_config.NumberColumn(probability_label, format="percent"),
        "Полнота данных": st.column_config.NumberColumn("Полнота данных", format="percent"),
    }

    st.dataframe(df_show, use_container_width=True, hide_index=True, column_config=dataframe_config)

    if not df_filtered.empty:
        output = io.BytesIO()
        export_cols = [c for c in display_columns if c in df_filtered.columns]

        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            sheet_name = "Winback_Scoring" if is_churned else "Churn_Scoring"
            df_filtered[export_cols].rename(columns=rename_columns).to_excel(writer, index=False, sheet_name=sheet_name)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scoring_{client_type}_{ts}.xlsx"

        st.download_button(
            label="Скачать отчёт в Excel",
            data=output.getvalue(),
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


init_session_state()
sync_manual_features_from_widgets()

with st.sidebar:
    st.header("Настройки системы")

    mode_label = st.radio(
        "Режим скоринга",
        options=["ML-модель (CatBoost)", "Ручной скоринг"],
        index=0 if st.session_state["scoring_mode"] == "ml" else 1,
    )
    current_mode = "ml" if mode_label.startswith("ML") else "manual"

    # НОВЫЙ ПЕРЕКЛЮЧАТЕЛЬ ТИПА КЛИЕНТА
    client_label = st.radio(
        "Тип клиентов",
        options=["Сопровождаемые (Активные)", "Ушедшие (Возврат)"],
        index=0 if st.session_state["client_type"] == "active" else 1,
    )
    current_client_type = "active" if client_label.startswith("Соп") else "churned"

    changed = False
    if current_mode != st.session_state["scoring_mode"]:
        st.session_state["scoring_mode"] = current_mode
        changed = True

    if current_client_type != st.session_state["client_type"]:
        st.session_state["client_type"] = current_client_type
        # Меняем дефолтные признаки при смене типа клиента
        default_feats = DEFAULT_FEATURES_ACTIVE if current_client_type == "active" else DEFAULT_FEATURES_CHURNED
        st.session_state["manual_features"] = [f.copy() for f in default_feats]
        changed = True

    if changed:
        st.session_state["result_data"] = None
        st.rerun()

    st.divider()

    uploaded_file = st.file_uploader(
        "Загрузите Excel-файл",
        type=["xlsx"],
        key=f"excel_uploader_{st.session_state['uploader_version']}",
    )

    if uploaded_file is not None:
        st.success(f"Выбран файл: {uploaded_file.name}")

        button_label = (
            "Рассчитать ML-скоринг"
            if current_mode == "ml"
            else "Рассчитать ручной скоринг"
        )

        invalid_manual_weights = current_mode == "manual" and not manual_weights_are_valid()

        if invalid_manual_weights:
            tg = "Риска" if current_client_type == "active" else "Возвращаемости"
            st.warning(
                f"Сумма весов ({tg}) должна быть равна 1.00. "
                f"Текущая сумма: {get_group_total('risk' if current_client_type == 'active' else 'value'):.3f}. "
                "Исправьте веса или нажмите автокоррекцию."
            )

        calculate_clicked = st.button(
            button_label,
            type="primary",
            disabled=invalid_manual_weights,
            help=("Сначала приведите сумму весов к 1.0" if invalid_manual_weights else None),
        )

        if calculate_clicked:
            with st.spinner("Обработка данных..."):
                result = send_scoring_request(
                    uploaded_file,
                    current_mode,
                    current_client_type,
                )

            if result is not None:
                st.session_state["result_data"] = result
                st.success("Расчёт завершён.")
                st.rerun()

    st.button("Очистить результаты", on_click=clear_results_callback)

if st.session_state["result_data"] is not None:
    render_dashboard(
        st.session_state["result_data"],
        st.session_state["scoring_mode"],
        st.session_state["client_type"],
    )
elif st.session_state["scoring_mode"] == "manual":
    render_manual_scoring_constructor()
else:
    cl_type_text = "активных" if st.session_state["client_type"] == "active" else "ушедших"
    st.info(
        f"Загрузите Excel-файл в боковой панели и нажмите «Рассчитать ML-скоринг» для {cl_type_text} клиентов."
    )