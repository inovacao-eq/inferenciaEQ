import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from io import StringIO
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StatInfer · Inferências Estatísticas",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #0f1117; }

h1, h2, h3 { font-family: 'Inter', sans-serif; font-weight: 700; }

.metric-card {
    background: linear-gradient(135deg, #1e2130 0%, #252a3d 100%);
    border: 1px solid #2e3350;
    border-radius: 12px;
    padding: 18px 22px;
    margin: 6px 0;
}

.result-box {
    background: #141722;
    border-left: 4px solid #5b8dee;
    border-radius: 0 10px 10px 0;
    padding: 16px 20px;
    margin: 12px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.88rem;
}

.result-box.success { border-left-color: #2ecc71; }
.result-box.danger  { border-left-color: #e74c3c; }
.result-box.warning { border-left-color: #f39c12; }

.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    margin: 2px 4px;
}
.badge-blue   { background: #1a3a6b; color: #5b8dee; }
.badge-green  { background: #0f3d2e; color: #2ecc71; }
.badge-red    { background: #3d0f0f; color: #e74c3c; }
.badge-yellow { background: #3d2e0f; color: #f39c12; }

.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #a0a8c0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 24px 0 12px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #2e3350;
}

.stButton > button {
    background: linear-gradient(135deg, #3b5bdb, #5b8dee);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 24px;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

.stSelectbox > div, .stMultiSelect > div { border-radius: 8px !important; }

[data-testid="stSidebar"] {
    background: #0a0d14;
    border-right: 1px solid #1e2130;
}

.interpretation {
    background: #1a1e2e;
    border-radius: 10px;
    padding: 16px 20px;
    margin-top: 12px;
    font-size: 0.92rem;
    line-height: 1.65;
    color: #c8cfe0;
}
</style>
""", unsafe_allow_html=True)

# ── Matplotlib dark theme ───────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#141722",
    "axes.facecolor":   "#1e2130",
    "axes.edgecolor":   "#2e3350",
    "axes.labelcolor":  "#a0a8c0",
    "text.color":       "#c8cfe0",
    "xtick.color":      "#a0a8c0",
    "ytick.color":      "#a0a8c0",
    "grid.color":       "#2e3350",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "font.family":      "sans-serif",
})

BLUE   = "#5b8dee"
GREEN  = "#2ecc71"
RED    = "#e74c3c"
YELLOW = "#f39c12"
PURPLE = "#9b59b6"

# ── Helpers ─────────────────────────────────────────────────────────────────────
def fmt(val, decimals=4):
    if isinstance(val, float):
        if abs(val) < 1e-4 and val != 0:
            return f"{val:.4e}"
        return f"{val:.{decimals}f}"
    return str(val)

def significance_badge(p, alpha=0.05):
    if p < 0.001:
        return f'<span class="badge badge-red">p &lt; 0.001 ✦✦✦</span>'
    elif p < 0.01:
        return f'<span class="badge badge-red">p &lt; 0.01 ✦✦</span>'
    elif p < alpha:
        return f'<span class="badge badge-yellow">p &lt; {alpha} ✦</span>'
    else:
        return f'<span class="badge badge-blue">p ≥ {alpha} ns</span>'

def interpret_p(p, test_name, alpha=0.05):
    if p < alpha:
        return f"✅ **Resultado significativo** (p = {fmt(p)}). Há evidência suficiente para **rejeitar H₀** ao nível de significância α = {alpha}."
    else:
        return f"⚪ **Resultado não significativo** (p = {fmt(p)}). **Não há evidência suficiente** para rejeitar H₀ ao nível α = {alpha}."

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 StatInfer")
    st.markdown("<small style='color:#5b6480'>Inferências estatísticas interativas</small>", unsafe_allow_html=True)
    st.divider()

    st.markdown("### Carregar dados")
    data_source = st.radio("Fonte dos dados", ["Upload CSV/Excel", "Colar dados (CSV)", "Dataset de exemplo"])

    df = None

    if data_source == "Upload CSV/Excel":
        uploaded = st.file_uploader("Selecione o arquivo", type=["csv", "xlsx", "xls"])
        if uploaded:
            try:
                if uploaded.name.endswith((".xlsx", ".xls")):
                    df = pd.read_excel(uploaded)
                else:
                    sep = st.selectbox("Separador", [",", ";", "\t", "|"], index=1)
                    df = pd.read_csv(uploaded, sep=sep)
                st.success(f"✓ {df.shape[0]} linhas × {df.shape[1]} colunas")
            except Exception as e:
                st.error(f"Erro ao carregar: {e}")

    elif data_source == "Colar dados (CSV)":
        raw = st.text_area("Cole o CSV aqui", height=160, placeholder="col1,col2\n1.2,3.4\n...")
        sep = st.selectbox("Separador", [",", ";", "\t", "|"], index=0)
        if raw.strip():
            try:
                df = pd.read_csv(StringIO(raw), sep=sep)
                st.success(f"✓ {df.shape[0]} linhas × {df.shape[1]} colunas")
            except Exception as e:
                st.error(f"Erro: {e}")

    else:  # Exemplo
        example = st.selectbox("Dataset", ["Iris (Fisher)", "Presença e Nota", "A/B Test Clicks"])
        if example == "Iris (Fisher)":
            from sklearn.datasets import load_iris
            iris = load_iris(as_frame=True)
            df = iris.frame
            df["species"] = df["target"].map({0:"setosa",1:"versicolor",2:"virginica"})
        elif example == "Presença e Nota":
            rng = np.random.default_rng(42)
            n = 80
            presença = rng.integers(60, 100, n).astype(float)
            nota = 4 + 0.06*presença + rng.normal(0, 1, n)
            nota = np.clip(nota, 0, 10)
            grupo = np.where(presença >= 80, "Alta", "Baixa")
            df = pd.DataFrame({"presença_pct": presença, "nota_final": nota, "grupo_presença": grupo})
        else:  # A/B
            rng = np.random.default_rng(7)
            n = 200
            grupo = np.repeat(["Controle","Tratamento"], n)
            cliques = np.concatenate([rng.binomial(1, 0.12, n), rng.binomial(1, 0.18, n)])
            tempo = np.concatenate([rng.normal(45, 12, n), rng.normal(38, 10, n)])
            df = pd.DataFrame({"grupo": grupo, "clique": cliques, "tempo_seg": tempo})
        st.success(f"✓ Dataset: **{example}** — {df.shape[0]}×{df.shape[1]}")

    if df is not None:
        st.divider()
        alpha = st.slider("Nível de significância (α)", 0.01, 0.10, 0.05, 0.01, format="%.2f")

# ── Main ────────────────────────────────────────────────────────────────────────
st.markdown("# 📊 StatInfer")
st.markdown("<p style='color:#5b6480;margin-top:-10px'>Plataforma de Inferências Estatísticas Interativas</p>", unsafe_allow_html=True)

if df is None:
    st.info("👈 Carregue seus dados na barra lateral para começar.")
    st.stop()

# Tabs
tab_eda, tab_norm, tab_ttest, tab_anova, tab_chi, tab_corr, tab_reg = st.tabs([
    "🔍 Explorar", "📐 Normalidade", "🔬 Testes t", "📊 ANOVA", "χ² Qui-Quadrado", "🔗 Correlação", "📈 Regressão"
])

num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()

# ── TAB 1 — EDA ─────────────────────────────────────────────────────────────────
with tab_eda:
    st.markdown("### Visão Geral dos Dados")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Linhas", df.shape[0])
    c2.metric("Colunas", df.shape[1])
    c3.metric("Numéricas", len(num_cols))
    c4.metric("Categóricas", len(cat_cols))

    st.dataframe(df.head(50), use_container_width=True, height=260)

    if num_cols:
        st.markdown('<div class="section-title">Estatísticas Descritivas</div>', unsafe_allow_html=True)
        desc = df[num_cols].describe().T
        desc["cv%"] = (desc["std"] / desc["mean"].abs() * 100).round(2)
        desc["skewness"] = df[num_cols].skew().round(4)
        desc["kurtosis"] = df[num_cols].kurt().round(4)
        st.dataframe(desc.style.format("{:.4f}", na_rep="—"), use_container_width=True)

        st.markdown('<div class="section-title">Distribuições</div>', unsafe_allow_html=True)
        sel_cols = st.multiselect("Colunas para visualizar", num_cols, default=num_cols[:min(4,len(num_cols))])
        if sel_cols:
            n_plots = len(sel_cols)
            ncols = min(n_plots, 3)
            nrows = (n_plots + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows))
            axes = np.array(axes).flatten() if n_plots > 1 else [axes]
            for i, col in enumerate(sel_cols):
                ax = axes[i]
                data = df[col].dropna()
                ax.hist(data, bins="auto", color=BLUE, alpha=0.7, edgecolor="#1e2130")
                ax2 = ax.twinx()
                sns.kdeplot(data, ax=ax2, color=YELLOW, linewidth=2)
                ax2.set_ylabel("")
                ax2.tick_params(left=False, right=False, labelleft=False, labelright=False)
                ax.set_title(col, fontweight="bold", color="#c8cfe0")
                ax.set_xlabel("Valor"); ax.set_ylabel("Frequência")
                ax.grid(True)
            for j in range(i+1, len(axes)):
                axes[j].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig); plt.close(fig)

    if cat_cols:
        st.markdown('<div class="section-title">Variáveis Categóricas</div>', unsafe_allow_html=True)
        cat_sel = st.selectbox("Coluna", cat_cols)
        vc = df[cat_sel].value_counts()
        fig, ax = plt.subplots(figsize=(8, 3.5))
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(vc)))[::-1]
        ax.barh(vc.index.astype(str), vc.values, color=colors)
        ax.set_xlabel("Contagem"); ax.set_title(cat_sel, fontweight="bold", color="#c8cfe0")
        ax.grid(True, axis="x")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        st.markdown('<div class="section-title">Valores Ausentes</div>', unsafe_allow_html=True)
        mv = missing[missing > 0].reset_index()
        mv.columns = ["Coluna", "Ausentes"]
        mv["% Total"] = (mv["Ausentes"] / len(df) * 100).round(2)
        st.dataframe(mv, use_container_width=True)

# ── TAB 2 — NORMALIDADE ─────────────────────────────────────────────────────────
with tab_norm:
    st.markdown("### Testes de Normalidade")
    if not num_cols:
        st.warning("Nenhuma coluna numérica encontrada."); st.stop()

    col = st.selectbox("Variável", num_cols)
    data = df[col].dropna()

    c1, c2 = st.columns([3, 2])
    with c1:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        # Histogram + KDE
        ax = axes[0]
        ax.hist(data, bins="auto", color=BLUE, alpha=0.65, density=True, edgecolor="#1e2130", label="Dados")
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 200)
        mu, sigma = data.mean(), data.std()
        ax.plot(x, stats.norm.pdf(x, mu, sigma), color=YELLOW, lw=2, label="Normal teórica")
        ax.legend(fontsize=9); ax.set_title(f"Histograma — {col}", fontweight="bold", color="#c8cfe0")
        ax.grid(True)
        # Q-Q Plot
        ax2 = axes[1]
        qq = stats.probplot(data, dist="norm")
        theoretical_q, ordered_v = qq[0]
        slope, intercept, r = qq[1]
        ax2.scatter(theoretical_q, ordered_v, color=BLUE, s=20, alpha=0.7)
        ax2.plot(theoretical_q, slope*theoretical_q + intercept, color=RED, lw=2)
        ax2.set_title("Q-Q Plot", fontweight="bold", color="#c8cfe0")
        ax2.set_xlabel("Quantis teóricos"); ax2.set_ylabel("Quantis da amostra")
        ax2.grid(True)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    with c2:
        st.markdown('<div class="section-title">Resultados dos Testes</div>', unsafe_allow_html=True)

        tests = {}
        # Shapiro-Wilk (n <= 5000)
        if len(data) <= 5000:
            stat, p = stats.shapiro(data)
            tests["Shapiro-Wilk"] = (stat, p)
        # Kolmogorov-Smirnov
        stat_ks, p_ks = stats.kstest(data, "norm", args=(data.mean(), data.std()))
        tests["Kolmogorov-Smirnov"] = (stat_ks, p_ks)
        # D'Agostino-Pearson
        if len(data) >= 8:
            stat_da, p_da = stats.normaltest(data)
            tests["D'Agostino-Pearson"] = (stat_da, p_da)
        # Anderson-Darling
        res_ad = stats.anderson(data, dist="norm")
        sig_levels = res_ad.significance_level
        crits = res_ad.critical_values
        idx_5 = list(sig_levels).index(5) if 5 in sig_levels else 2
        p_ad_approx = "< 0.05" if res_ad.statistic > crits[idx_5] else "≥ 0.05"

        for name, (stat_v, p_v) in tests.items():
            is_normal = p_v >= alpha
            box_cls = "success" if is_normal else "danger"
            icon = "✅" if is_normal else "❌"
            st.markdown(f"""
            <div class="result-box {box_cls}">
                <strong>{icon} {name}</strong><br>
                Estatística: <code>{fmt(stat_v)}</code>&nbsp;&nbsp;
                p-valor: <code>{fmt(p_v)}</code><br>
                {significance_badge(p_v, alpha)}
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="result-box">
            <strong>Anderson-Darling</strong><br>
            Estatística: <code>{fmt(res_ad.statistic)}</code>&nbsp;&nbsp;
            p-valor: <code>{p_ad_approx}</code>
        </div>
        """, unsafe_allow_html=True)

        norm_count = sum(1 for _, p_v in tests.values() if p_v >= alpha)
        total = len(tests)
        st.markdown(f"""
        <div class="interpretation">
            <b>Interpretação geral:</b><br>
            {norm_count}/{total} testes paramétricos indicam normalidade.<br><br>
            {"🟢 A distribuição é compatível com normalidade na maioria dos testes." if norm_count > total//2
              else "🔴 A distribuição não segue normalidade na maioria dos testes. Considere testes não-paramétricos."}
        </div>
        """, unsafe_allow_html=True)

# ── TAB 3 — TESTES t ────────────────────────────────────────────────────────────
with tab_ttest:
    st.markdown("### Testes t")
    test_type = st.radio("Tipo de teste", ["t Uma Amostra", "t Duas Amostras Independentes", "t Pareado (Wilcoxon)"], horizontal=True)

    if test_type == "t Uma Amostra":
        col = st.selectbox("Variável", num_cols)
        mu0 = st.number_input("Valor hipotético (H₀: μ = ?)", value=0.0)
        alt = st.selectbox("H₁", ["two-sided", "greater", "less"])
        if st.button("Executar Teste t Uma Amostra"):
            data = df[col].dropna()
            stat, p = stats.ttest_1samp(data, mu0, alternative=alt)
            ci = stats.t.interval(1-alpha, df=len(data)-1, loc=data.mean(), scale=stats.sem(data))
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="result-box">
                    <b>Resultado</b><br>
                    n = {len(data)}&nbsp;&nbsp; x̄ = {fmt(data.mean())}&nbsp;&nbsp; s = {fmt(data.std())}<br>
                    t = {fmt(stat)}&nbsp;&nbsp; gl = {len(data)-1}<br>
                    p-valor = {fmt(p)}&nbsp;&nbsp; {significance_badge(p, alpha)}
                    <br>IC {int((1-alpha)*100)}%: [{fmt(ci[0])}, {fmt(ci[1])}]
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f'<div class="interpretation">{interpret_p(p, "t", alpha)}</div>', unsafe_allow_html=True)
            with c2:
                fig, ax = plt.subplots(figsize=(5, 3.5))
                ax.hist(data, bins="auto", color=BLUE, alpha=0.65, edgecolor="#1e2130")
                ax.axvline(data.mean(), color=GREEN, lw=2, label=f"x̄ = {fmt(data.mean())}")
                ax.axvline(mu0, color=RED, lw=2, linestyle="--", label=f"μ₀ = {mu0}")
                ax.legend(fontsize=9); ax.grid(True)
                ax.set_title(f"Distribuição de {col}", color="#c8cfe0", fontweight="bold")
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    elif test_type == "t Duas Amostras Independentes":
        col = st.selectbox("Variável numérica", num_cols)
        grp_col = st.selectbox("Variável de grupo", cat_cols if cat_cols else num_cols)
        if grp_col:
            grupos = df[grp_col].dropna().unique()
            if len(grupos) >= 2:
                g1, g2 = st.selectbox("Grupo 1", grupos), st.selectbox("Grupo 2", [g for g in grupos if g != grupos[0]], index=0 if len(grupos)>1 else 0)
                eq_var = st.checkbox("Assumir variâncias iguais (Student)", value=False)
                if st.button("Executar Teste t Independente"):
                    d1 = df[df[grp_col]==g1][col].dropna()
                    d2 = df[df[grp_col]==g2][col].dropna()
                    # Levene test
                    lev_stat, lev_p = stats.levene(d1, d2)
                    stat, p = stats.ttest_ind(d1, d2, equal_var=eq_var)
                    cohen_d = (d1.mean()-d2.mean()) / np.sqrt(((len(d1)-1)*d1.std()**2 + (len(d2)-1)*d2.std()**2)/(len(d1)+len(d2)-2))

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"""
                        <div class="result-box">
                            <b>Teste de Levene (homocedasticidade)</b><br>
                            F = {fmt(lev_stat)}&nbsp; p = {fmt(lev_p)}&nbsp; {significance_badge(lev_p, alpha)}
                        </div>
                        <div class="result-box">
                            <b>Resultado: {"Student" if eq_var else "Welch"}</b><br>
                            {g1}: n={len(d1)}, x̄={fmt(d1.mean())}, s={fmt(d1.std())}<br>
                            {g2}: n={len(d2)}, x̄={fmt(d2.mean())}, s={fmt(d2.std())}<br>
                            t = {fmt(stat)}&nbsp; p = {fmt(p)}&nbsp; {significance_badge(p, alpha)}<br>
                            Cohen's d = {fmt(cohen_d)} {"(grande)" if abs(cohen_d)>0.8 else "(médio)" if abs(cohen_d)>0.5 else "(pequeno)"}
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f'<div class="interpretation">{interpret_p(p, "t", alpha)}</div>', unsafe_allow_html=True)
                    with c2:
                        fig, ax = plt.subplots(figsize=(5,4))
                        data_box = [d1.values, d2.values]
                        bp = ax.boxplot(data_box, patch_artist=True, widths=0.5,
                                        medianprops=dict(color=YELLOW, lw=2))
                        colors_box = [BLUE, GREEN]
                        for patch, c in zip(bp['boxes'], colors_box):
                            patch.set_facecolor(c); patch.set_alpha(0.6)
                        ax.set_xticklabels([str(g1), str(g2)])
                        ax.set_ylabel(col); ax.grid(True, axis="y")
                        ax.set_title("Comparação entre grupos", color="#c8cfe0", fontweight="bold")
                        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    else:  # Pareado
        col1 = st.selectbox("Variável 1 (antes)", num_cols)
        col2 = st.selectbox("Variável 2 (depois)", [c for c in num_cols if c != col1] or num_cols)
        if st.button("Executar Wilcoxon"):
            d1 = df[col1].dropna()
            d2 = df[col2].dropna()
            n = min(len(d1), len(d2))
            d1, d2 = d1[:n], d2[:n]
            stat, p = stats.wilcoxon(d1, d2)
            diff = d2 - d1
            st.markdown(f"""
            <div class="result-box">
                <b>Teste de Wilcoxon (Sinal com Postos)</b><br>
                n = {n}&nbsp; W = {fmt(stat)}&nbsp; p = {fmt(p)}&nbsp; {significance_badge(p, alpha)}<br>
                Mediana das diferenças: {fmt(diff.median())}
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f'<div class="interpretation">{interpret_p(p, "Wilcoxon", alpha)}</div>', unsafe_allow_html=True)

# ── TAB 4 — ANOVA ───────────────────────────────────────────────────────────────
with tab_anova:
    st.markdown("### ANOVA / Kruskal-Wallis")
    if not cat_cols:
        st.warning("Nenhuma variável categórica disponível."); st.stop()

    dep = st.selectbox("Variável dependente (numérica)", num_cols)
    grp = st.selectbox("Fator (grupos)", cat_cols)
    method = st.radio("Método", ["ANOVA paramétrica (F)", "Kruskal-Wallis (não-paramétrico)"], horizontal=True)

    if st.button("Executar Análise"):
        grupos = df[grp].dropna().unique()
        samples = [df[df[grp]==g][dep].dropna().values for g in grupos]
        samples = [s for s in samples if len(s) > 1]

        if len(samples) < 2:
            st.warning("É necessário ao menos 2 grupos com dados."); st.stop()

        if "ANOVA" in method:
            stat, p = stats.f_oneway(*samples)
            test_name = "ANOVA"
            stat_label = "F"
        else:
            stat, p = stats.kruskal(*samples)
            test_name = "Kruskal-Wallis"
            stat_label = "H"

        c1, c2 = st.columns([2, 3])
        with c1:
            desc_rows = []
            for g, s in zip(grupos, samples):
                desc_rows.append({"Grupo": g, "n": len(s), "Média": round(np.mean(s),4),
                                   "Mediana": round(np.median(s),4), "DP": round(np.std(s),4)})
            st.dataframe(pd.DataFrame(desc_rows), use_container_width=True, hide_index=True)

            st.markdown(f"""
            <div class="result-box">
                <b>{test_name}</b><br>
                {stat_label} = {fmt(stat)}&nbsp; p = {fmt(p)}&nbsp; {significance_badge(p, alpha)}
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f'<div class="interpretation">{interpret_p(p, test_name, alpha)}</div>', unsafe_allow_html=True)

            # Post-hoc (Tukey)
            if p < alpha and "ANOVA" in method:
                try:
                    from scipy.stats import tukey_hsd
                    th = tukey_hsd(*samples)
                    st.markdown("**Post-hoc: Tukey HSD**")
                    rows = []
                    for i, gi in enumerate(grupos[:len(samples)]):
                        for j, gj in enumerate(grupos[:len(samples)]):
                            if j > i:
                                pval = th.pvalue[i][j]
                                rows.append({"Grupo A": gi, "Grupo B": gj, "p-valor": round(pval,4),
                                             "Sig.": "✓" if pval < alpha else ""})
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                except Exception:
                    pass

        with c2:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            palette = [BLUE, GREEN, YELLOW, PURPLE, RED, "#1abc9c", "#e67e22"]
            bps = ax.boxplot(samples, patch_artist=True, widths=0.5,
                             medianprops=dict(color="white", lw=2))
            for bp_box, c in zip(bps["boxes"], palette):
                bp_box.set_facecolor(c); bp_box.set_alpha(0.65)
            for i, (g, s) in enumerate(zip(grupos, samples)):
                x = np.random.normal(i+1, 0.07, size=len(s))
                ax.scatter(x, s, alpha=0.4, s=18, color="white")
            ax.set_xticklabels([str(g) for g in grupos], rotation=30, ha="right")
            ax.set_ylabel(dep); ax.grid(True, axis="y")
            ax.set_title(f"{dep} por {grp}", color="#c8cfe0", fontweight="bold")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

# ── TAB 5 — CHI-SQUARE ──────────────────────────────────────────────────────────
with tab_chi:
    st.markdown("### Teste Qui-Quadrado (χ²)")
    if len(cat_cols) < 1:
        st.warning("Nenhuma variável categórica disponível."); st.stop()

    col1 = st.selectbox("Variável 1", cat_cols, key="chi1")
    col2 = st.selectbox("Variável 2", [c for c in cat_cols if c != col1] or cat_cols, key="chi2")

    if st.button("Executar Qui-Quadrado"):
        ct = pd.crosstab(df[col1], df[col2])
        chi2, p, dof, expected = stats.chi2_contingency(ct)
        n = ct.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(ct.shape)-1)))

        c1, c2 = st.columns([2, 3])
        with c1:
            st.markdown("**Tabela de Contingência**")
            st.dataframe(ct, use_container_width=True)
            st.markdown(f"""
            <div class="result-box">
                <b>Resultado</b><br>
                χ² = {fmt(chi2)}&nbsp; gl = {dof}&nbsp; p = {fmt(p)}<br>
                {significance_badge(p, alpha)}<br>
                Cramér's V = {fmt(cramers_v)} — Efeito: {"forte" if cramers_v>0.5 else "moderado" if cramers_v>0.3 else "fraco"}
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f'<div class="interpretation">{interpret_p(p, "Qui-Quadrado", alpha)}</div>', unsafe_allow_html=True)
        with c2:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
            ct_pct.plot(kind="bar", ax=ax, colormap="cool", edgecolor="#1e2130", width=0.7)
            ax.set_ylabel("% dentro do grupo"); ax.set_xlabel(col1)
            ax.set_title(f"Distribuição de {col2} por {col1}", color="#c8cfe0", fontweight="bold")
            ax.legend(title=col2, fontsize=9, title_fontsize=9)
            ax.grid(True, axis="y"); plt.xticks(rotation=30, ha="right")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

# ── TAB 6 — CORRELAÇÃO ──────────────────────────────────────────────────────────
with tab_corr:
    st.markdown("### Análise de Correlação")
    if len(num_cols) < 2:
        st.warning("Necessário ao menos 2 colunas numéricas."); st.stop()

    method_corr = st.radio("Método", ["Pearson", "Spearman", "Kendall"], horizontal=True)
    sel = st.multiselect("Colunas", num_cols, default=num_cols[:min(6, len(num_cols))])

    if len(sel) >= 2:
        corr_df = df[sel].dropna()
        if method_corr == "Pearson":
            corr_mat = corr_df.corr(method="pearson")
        elif method_corr == "Spearman":
            corr_mat = corr_df.corr(method="spearman")
        else:
            corr_mat = corr_df.corr(method="kendall")

        c1, c2 = st.columns([3, 2])
        with c1:
            fig, ax = plt.subplots(figsize=(8, 6))
            mask = np.triu(np.ones_like(corr_mat, dtype=bool))
            cmap = sns.diverging_palette(220, 10, as_cmap=True)
            sns.heatmap(corr_mat, mask=mask, annot=True, fmt=".2f", cmap=cmap,
                        center=0, ax=ax, linewidths=0.5, linecolor="#1e2130",
                        annot_kws={"size": 10, "weight": "bold"},
                        cbar_kws={"shrink": 0.8})
            ax.set_title(f"Matriz de Correlação ({method_corr})", color="#c8cfe0", fontweight="bold")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        with c2:
            st.markdown('<div class="section-title">Correlações Significativas</div>', unsafe_allow_html=True)
            rows = []
            cols_l = list(sel)
            for i in range(len(cols_l)):
                for j in range(i+1, len(cols_l)):
                    r = corr_mat.iloc[i, j]
                    n = len(corr_df)
                    t_stat = r * np.sqrt((n-2)/(1-r**2)) if abs(r) < 1 else np.inf
                    pval = 2*stats.t.sf(abs(t_stat), df=n-2) if np.isfinite(t_stat) else 0.0
                    rows.append({"Var 1": cols_l[i], "Var 2": cols_l[j], "r": round(r,4), "p": round(pval,4)})
            sig_df = pd.DataFrame(rows).sort_values("r", key=abs, ascending=False)
            sig_df["Sig."] = sig_df["p"].apply(lambda x: "✓" if x < alpha else "")
            st.dataframe(sig_df, use_container_width=True, hide_index=True)

        if len(sel) == 2:
            st.markdown('<div class="section-title">Scatter Plot</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.scatter(corr_df[sel[0]], corr_df[sel[1]], color=BLUE, alpha=0.5, s=30)
            m, b, r, p_reg, _ = stats.linregress(corr_df[sel[0]], corr_df[sel[1]])
            x_line = np.linspace(corr_df[sel[0]].min(), corr_df[sel[0]].max(), 100)
            ax.plot(x_line, m*x_line+b, color=RED, lw=2, label=f"y={fmt(m)}x+{fmt(b)}")
            ax.set_xlabel(sel[0]); ax.set_ylabel(sel[1])
            ax.set_title(f"r={fmt(corr_mat.iloc[0,1])} ({method_corr})", color="#c8cfe0", fontweight="bold")
            ax.legend(); ax.grid(True)
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

# ── TAB 7 — REGRESSÃO ───────────────────────────────────────────────────────────
with tab_reg:
    st.markdown("### Regressão Linear")
    if len(num_cols) < 2:
        st.warning("Necessário ao menos 2 colunas numéricas."); st.stop()

    dep_reg = st.selectbox("Variável dependente (Y)", num_cols, key="dep_reg")
    ind_reg = st.multiselect("Variáveis independentes (X)", [c for c in num_cols if c != dep_reg], default=[c for c in num_cols if c != dep_reg][:1])

    if ind_reg and st.button("Executar Regressão"):
        from scipy.stats import t as t_dist

        sub = df[[dep_reg] + ind_reg].dropna()
        Y = sub[dep_reg].values
        X_raw = sub[ind_reg].values
        X = np.column_stack([np.ones(len(X_raw)), X_raw])

        # OLS manual via numpy
        try:
            beta = np.linalg.lstsq(X, Y, rcond=None)[0]
            Y_hat = X @ beta
            resid = Y - Y_hat
            n, k = X.shape
            SSR = np.sum((Y_hat - Y.mean())**2)
            SSE = np.sum(resid**2)
            SST = np.sum((Y - Y.mean())**2)
            R2 = SSR / SST
            R2_adj = 1 - (1-R2)*(n-1)/(n-k)
            MSE = SSE / (n-k)
            RMSE = np.sqrt(MSE)
            F_stat = (SSR/(k-1)) / MSE
            p_F = 1 - stats.f.cdf(F_stat, k-1, n-k)

            # Coeficientes
            XtX_inv = np.linalg.pinv(X.T @ X)
            se = np.sqrt(np.diag(XtX_inv) * MSE)
            t_stats = beta / se
            p_coef = 2 * t_dist.sf(np.abs(t_stats), df=n-k)

            c1, c2 = st.columns([3, 2])
            with c1:
                coef_df = pd.DataFrame({
                    "Termo": ["Intercepto"] + ind_reg,
                    "β": [fmt(b) for b in beta],
                    "Erro Padrão": [fmt(s) for s in se],
                    "t": [fmt(t) for t in t_stats],
                    "p-valor": [fmt(p) for p in p_coef],
                    "Sig.": ["✓" if p < alpha else "" for p in p_coef]
                })
                st.dataframe(coef_df, use_container_width=True, hide_index=True)

                st.markdown(f"""
                <div class="result-box">
                    <b>Qualidade do Modelo</b><br>
                    R² = {fmt(R2)}&nbsp;&nbsp; R² ajustado = {fmt(R2_adj)}<br>
                    RMSE = {fmt(RMSE)}&nbsp;&nbsp; n = {n}<br>
                    F({k-1},{n-k}) = {fmt(F_stat)}&nbsp; p = {fmt(p_F)}&nbsp; {significance_badge(p_F, alpha)}
                </div>
                """, unsafe_allow_html=True)

                fit_qual = "excelente (R²>0.9)" if R2>0.9 else "bom (R²>0.7)" if R2>0.7 else "moderado (R²>0.5)" if R2>0.5 else "fraco (R²≤0.5)"
                st.markdown(f"""
                <div class="interpretation">
                    O modelo explica <b>{fmt(R2*100, 1)}%</b> da variância de <em>{dep_reg}</em>. Ajuste <b>{fit_qual}</b>.<br><br>
                    {interpret_p(p_F, "Regressão", alpha)}
                </div>
                """, unsafe_allow_html=True)

            with c2:
                fig, axes = plt.subplots(2, 1, figsize=(5, 7))
                # Fitted vs Actual
                ax = axes[0]
                ax.scatter(Y_hat, Y, color=BLUE, alpha=0.5, s=25)
                mn, mx = min(Y.min(), Y_hat.min()), max(Y.max(), Y_hat.max())
                ax.plot([mn, mx], [mn, mx], color=RED, lw=1.5, linestyle="--")
                ax.set_xlabel("Valores Ajustados"); ax.set_ylabel("Valores Reais")
                ax.set_title("Ajustado vs Real", color="#c8cfe0", fontweight="bold"); ax.grid(True)
                # Resíduos
                ax2 = axes[1]
                ax2.scatter(Y_hat, resid, color=GREEN, alpha=0.5, s=25)
                ax2.axhline(0, color=RED, lw=1.5, linestyle="--")
                ax2.set_xlabel("Valores Ajustados"); ax2.set_ylabel("Resíduos")
                ax2.set_title("Resíduos vs Ajustado", color="#c8cfe0", fontweight="bold"); ax2.grid(True)
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        except Exception as e:
            st.error(f"Erro na regressão: {e}")

# ── Footer ───────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<small style='color:#3a405a'>StatInfer · Construído com Python, Streamlit, SciPy & Seaborn</small>",
    unsafe_allow_html=True
)
