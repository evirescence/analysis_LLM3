import io
import os
import sys
import base64
import traceback
import contextlib
import textwrap
from pathlib import Path

import anthropic
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="Анализатор данных по ИИ",
    layout="wide",
)


st.markdown("""
<style>
  .main { background: #0f1117; }
  .stApp { background: #ffffff; }
  h1 { color: #000000 !important; }
  .tool-box {
      background: #000000;
      border-left: 3px solid #6366f1;
      border-radius: 6px;
      padding: 10px 14px;
      font-family: monospace;
      font-size: 12px;
      color: white;
      margin: 6px 0;
  }
  .insight-box {
      background: #1e293b;
      border-radius: 10px;
      padding: 18px 22px;
      color: white;
      line-height: 1.7;
  }
</style>
""", unsafe_allow_html=True)

MODEL        = "claude-opus-4-5"
MAX_TOKENS   = 4096
MAX_LOOPS    = 8
PLOT_DPI     = 130

SYSTEM_PROMPT = """You are an expert data analyst. You have access to a Python code interpreter tool called `run_python`.
RULES:
- Always use `run_python` to perform ANY analysis, statistics, or chart creation. Never guess or hallucinate numbers.
- The dataframe is already loaded as `df` in the execution environment.
- For charts: use matplotlib/seaborn/plotly. Call `plt.tight_layout()` and `plt.savefig('plot.png', dpi=130, bbox_inches='tight')` then `plt.close()`. The image will be captured automatically.
- After all tool calls are done, write a concise markdown summary with: key findings, anomalies, recommendations. Use emojis for readability.
- Respond in the same language the user writes in.
- Keep code clean, commented, and efficient."""

TOOLS = [
    {
        "name": "run_python",
        "description": (
            "Execute Python code in a sandbox that has access to the uploaded dataframe as `df`. "
            "Available libraries: pandas, numpy, matplotlib, seaborn, plotly, scipy, sklearn. "
            "To save a chart write: plt.savefig('plot.png', dpi=130, bbox_inches='tight'); plt.close(). "
            "Return values via print(). The tool returns stdout + any generated plot image."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Valid Python code to execute."
                }
            },
            "required": ["code"]
        }
    }
]

def execute_python(code: str, df: pd.DataFrame) -> tuple[str, bytes | None]:
    plot_path = Path("/tmp/plot.png")
    if plot_path.exists():
        plot_path.unlink()

    stdout_buf = io.StringIO()
    ns = {
        "df": df.copy(),
        "pd": pd,
        "plt": plt,
    }
    for lib_name, alias in [("numpy","np"),("seaborn","sns"),
                             ("scipy","scipy"),("sklearn","sklearn")]:
        try:
            import importlib
            ns[alias] = importlib.import_module(lib_name)
        except ImportError:
            pass

    error_text = ""
    try:
        with contextlib.redirect_stdout(stdout_buf):
            exec(compile(code, "<llm_code>", "exec"), ns)  # noqa: S102
    except Exception:
        error_text = traceback.format_exc()

    output = stdout_buf.getvalue()
    if error_text:
        output += f"\n[ERROR]\n{error_text}"

    png_bytes = None
    if plot_path.exists():
        png_bytes = plot_path.read_bytes()
        plot_path.unlink()

    return output.strip(), png_bytes


def run_agent(client: anthropic.Anthropic, df: pd.DataFrame,
              user_question: str, progress_placeholder, log_placeholder):

    messages = [{"role": "user", "content": user_question}]
    images_collected: list[bytes] = []
    loop_count = 0

    while loop_count < MAX_LOOPS:
        loop_count += 1
        progress_placeholder.progress(
            min(loop_count / MAX_LOOPS, 0.95),
            text=f"Агент думает… шаг {loop_count}"
        )

        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            final_text = " ".join(
                b.text for b in response.content if hasattr(b, "text")
            )
            return final_text, images_collected

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            code = block.input.get("code", "")

            log_placeholder.markdown(
                f'<div class="tool-box"> <b>run_python</b> (шаг {loop_count})<br><br>'
                + "<br>".join(textwrap.wrap(code.replace("\n","↵ "), 120))
                + "</div>",
                unsafe_allow_html=True,
            )

            stdout, png = execute_python(code, df)
            if png:
                images_collected.append(png)

            result_content = stdout if stdout else "(no output)"

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_content,
            })

        messages.append({"role": "user", "content": tool_results})

    return "Достигнут лимит шагов агента.", images_collected


st.title("Анализатор данных по ИИ")
st.caption("Загрузи датасет, задай вопрос и агент сам напишет и выполнит код")

with st.sidebar:
    st.header("Настройки")
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Получить на console.anthropic.com",
    )
    st.divider()
    st.markdown("**Модель:** `claude-opus-4-5`")
    st.markdown("**Инструмент:** `run_python` (code interpreter)")
    st.markdown("**Агентный цикл:** до 8 шагов")
    st.divider()
    st.markdown("#### Примеры вопросов")
    example_questions = [
        "Сделай полный разведочный анализ датасета",
        "Найди корреляции между числовыми признаками и построй heatmap",
        "Определи выбросы и визуализируй их",
        "Покажи топ-10 значений по ключевым метрикам",
        "Построй распределения всех числовых колонок",
    ]
    for q in example_questions:
        if st.button(q, use_container_width=True):
            st.session_state["question_prefill"] = q

uploaded = st.file_uploader(
    "Загрузи CSV или Excel файл",
    type=["csv", "xlsx", "xls"],
    help="Максимальный размер — 200 МБ"
)

df = None
if uploaded:
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
        st.success(f"Загружено: **{uploaded.name}** — {df.shape[0]:,} строк × {df.shape[1]} столбцов")

        with st.expander("Предпросмотр данных (первые 5 строк)"):
            st.dataframe(df.head(), use_container_width=True)

        with st.expander("Базовая статистика"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Типы данных**")
                dtype_df = pd.DataFrame({
                    "Колонка": df.dtypes.index,
                    "Тип": df.dtypes.values.astype(str),
                    "Пропуски": df.isnull().sum().values,
                })
                st.dataframe(dtype_df, use_container_width=True, hide_index=True)
            with col2:
                st.markdown("**Числовые признаки**")
                st.dataframe(df.describe().T.round(3), use_container_width=True)

    except Exception as e:
        st.error(f"Ошибка при чтении файла: {e}")


prefill = st.session_state.pop("question_prefill", "")
question = st.text_area(
    "Что проанализировать?",
    value=prefill,
    placeholder="Например: сделай полный EDA и найди ключевые инсайты",
    height=90,
)

run_btn = st.button("Запустить анализ", type="primary",
                    disabled=(df is None or not question.strip() or not api_key))

if not api_key and run_btn:
    st.warning("Введите Anthropic API Key в боковой панели.")

if run_btn and df is not None and question.strip() and api_key:
    client = anthropic.Anthropic(api_key=api_key)

    st.divider()
    st.subheader("Работа агента")

    progress = st.empty()
    code_log = st.empty()

    with st.spinner("Агент анализирует данные…"):
        context = (
            f"Dataset info:\n"
            f"- Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
            f"- Columns: {list(df.columns)}\n"
            f"- Dtypes: {df.dtypes.to_dict()}\n\n"
            f"User question: {question}"
        )
        final_text, images = run_agent(client, df, context, progress, code_log)

    progress.progress(1.0, text="Анализ завершён!")

    st.divider()
    st.subheader("Графики")
    if images:
        cols = st.columns(min(len(images), 2))
        for i, img_bytes in enumerate(images):
            cols[i % 2].image(img_bytes, use_container_width=True)
    else:
        st.info("Графики не были сгенерированы для этого запроса.")

    st.divider()
    st.subheader("Инсайты и выводы")
    st.markdown(
        f'<div class="insight-box">{final_text}</div>',
        unsafe_allow_html=True,
    )

    # кнопка скачать отчёт
    report_md = f"# DataSage Report\n\n**Вопрос:** {question}\n\n---\n\n{final_text}"
    st.download_button(
        "Скачать отчёт (Markdown)",
        data=report_md.encode("utf-8"),
        file_name="datasage_report.md",
        mime="text/markdown",
    )

elif df is None:
    st.info("Загрузи файл выше чтобы начать.")