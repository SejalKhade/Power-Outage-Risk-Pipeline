"""
Power Outage Risk Dashboard
Sejal Khade | MS Data Science, UT Arlington
EIA-861 + NOAA Storm Events 2024 | 1,677 utilities
Maps: Folium (real interactive OpenStreetMap) — no plotly geo bugs
Charts: Plotly (choropleth, bar, violin, radar, heatmap, bubble)
"""
import warnings; warnings.filterwarnings("ignore")
import os, json, numpy as np, pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import folium
from folium.plugins import MarkerCluster
import gradio as gr

# ── Constants ──────────────────────────────────────────────────────
SC   = "IEEE_AllEvents_SAIDI_min_per_yr"
SC2  = "IEEE_AllEvents_SAIFI_times_per_yr"
BG   = "#0E1117"
BG2  = "#151B23"
GR   = "#2A3040"
TX   = "#FAFAFA"
RC   = {"High Risk":"#E74C3C","Medium Risk":"#F39C12","Low Risk":"#27AE60"}
RC_F = {"High Risk":"red",    "Medium Risk":"orange", "Low Risk":"green"}
HP   = ["#66C2A5","#FC8D62","#8DA0CB","#E78AC3",
        "#A6D854","#FFD92F","#E5C494","#B3B3B3",
        "#1F78B4","#33A02C","#E31A1C","#FF7F00"]
CENTS = {
    "AL":(32.81,-86.79),"AK":(61.37,-152.40),"AZ":(33.73,-111.43),
    "AR":(34.97,-92.37),"CA":(36.12,-119.68),"CO":(39.06,-105.31),
    "CT":(41.60,-72.76),"DE":(39.32,-75.51),"FL":(27.77,-81.69),
    "GA":(33.04,-83.64),"HI":(21.09,-157.50),"ID":(44.24,-114.48),
    "IL":(40.35,-88.99),"IN":(39.85,-86.26),"IA":(42.01,-93.21),
    "KS":(38.53,-96.73),"KY":(37.67,-84.67),"LA":(31.17,-91.87),
    "ME":(44.69,-69.38),"MD":(39.06,-76.80),"MA":(42.23,-71.53),
    "MI":(43.33,-84.54),"MN":(45.69,-93.90),"MS":(32.74,-89.68),
    "MO":(38.46,-92.29),"MT":(46.92,-110.45),"NE":(41.13,-98.27),
    "NV":(38.31,-117.06),"NH":(43.45,-71.56),"NJ":(40.30,-74.52),
    "NM":(34.84,-106.25),"NY":(42.17,-74.95),"NC":(35.63,-79.81),
    "ND":(47.53,-99.78),"OH":(40.39,-82.76),"OK":(35.57,-96.93),
    "OR":(44.57,-122.07),"PA":(40.59,-77.21),"RI":(41.68,-71.51),
    "SC":(33.86,-80.95),"SD":(44.30,-99.44),"TN":(35.75,-86.69),
    "TX":(31.05,-97.56),"UT":(40.15,-111.86),"VT":(44.05,-72.71),
    "VA":(37.77,-78.17),"WA":(47.40,-121.49),"WV":(38.49,-80.95),
    "WI":(44.27,-89.62),"WY":(42.76,-107.30),
}

# ── Helpers ────────────────────────────────────────────────────────
def ef(msg=""):
    fig = go.Figure()
    fig.add_annotation(text=str(msg) or "No data", x=0.5, y=0.5,
        showarrow=False, font=dict(color="#555", size=12))
    fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG,
        margin=dict(l=10,r=10,t=10,b=10))
    return fig

def lyt(fig, h=400, title=None):
    kw = dict(paper_bgcolor=BG, plot_bgcolor=BG2, font_color=TX,
        legend=dict(bgcolor="#1E2530", font=dict(color=TX,size=10)),
        margin=dict(l=20,r=20,t=55 if title else 30,b=20), height=h)
    if title: kw["title"] = dict(text=title, x=0.5, font=dict(size=13))
    fig.update_layout(**kw)
    fig.update_xaxes(gridcolor=GR, zerolinecolor=GR)
    fig.update_yaxes(gridcolor=GR, zerolinecolor=GR)
    return fig

def _map_html(height=460):
    """Empty dark map placeholder HTML."""
    return f"""<div style="height:{height}px;background:#0D1420;
        display:flex;align-items:center;justify-content:center;
        color:#555;font-size:14px;border-radius:8px;border:1px solid #2A3040;">
        Change any filter to load the map</div>"""

# ── Data ───────────────────────────────────────────────────────────
def load():
    path = "data/processed/utility_features.parquet"
    if os.path.exists(path):
        df = pd.read_parquet(path)
        print(f"✓ Real data: {df.shape}")
    else:
        print("⚠ Synthetic fallback")
        np.random.seed(42); st=list(CENTS.keys()); n=1677
        df = pd.DataFrame({
            "Utility Number":range(1,n+1),
            "Utility Name":[f"Utility {i:04d}" for i in range(1,n+1)],
            "State":np.random.choice(st,n),
            "Ownership":np.random.choice(
                ["Investor Owned","Cooperative","Municipal","Political Subdivision","Federal"],
                n,p=[.4,.3,.15,.1,.05]),
            "NERC Region":np.random.choice(["RFC","SERC","SPP","WECC","TRE","MRO","NPCC","FRCC"],n),
            "County_Count":np.random.randint(1,30,n),
            SC:np.random.exponential(300,n), SC2:np.random.exponential(1.3,n),
            "weather_event_count":np.random.poisson(45,n),
            "total_damage_usd":np.random.exponential(500_000,n),
            "INJURIES_DIRECT":np.random.poisson(.5,n),"DEATHS_DIRECT":np.random.poisson(.05,n),
            "human_impact_score":np.random.exponential(2,n),
        })
        df["saidi_rank_pct"]=df[SC].rank(pct=True); df["saifi_rank_pct"]=df[SC2].rank(pct=True)
        df["risk_score"]=(df["saidi_rank_pct"]+df["saifi_rank_pct"])/2
        q80,q50=df["risk_score"].quantile(.80),df["risk_score"].quantile(.50)
        df["risk_category"]=np.where(df["risk_score"]>=q80,"High Risk",
                            np.where(df["risk_score"]>=q50,"Medium Risk","Low Risk"))
        df["high_risk"]=(df["risk_score"]>=q80).astype(int)
        df["estimated_annual_loss_usd"]=(df[SC]/60)*df["County_Count"]*50_000*27
        df["nerc_sla_breach_risk"]=(df[SC]>150).astype(int)
    for c in ["estimated_annual_loss_usd","nerc_sla_breach_risk","risk_score",
              "risk_category","high_risk","weather_event_count","total_damage_usd","human_impact_score"]:
        if c not in df.columns: df[c]=0
    np.random.seed(0)
    df["lat"]=df["State"].map(lambda s:CENTS.get(s,(39.5,-98.35))[0])+np.random.uniform(-1.5,1.5,len(df))
    df["lon"]=df["State"].map(lambda s:CENTS.get(s,(39.5,-98.35))[1])+np.random.uniform(-2.0,2.0,len(df))
    return df

DF    = load()
TIERS = sorted(DF["risk_category"].unique().tolist())
UNAMES= sorted(DF["Utility Name"].dropna().unique().tolist())[:300]

def filt(state,own,nerc,tiers):
    d=DF.copy()
    if state and state!="All States":  d=d[d["State"]==state]
    if own   and own!="All Types":     d=d[d["Ownership"]==own]
    if nerc  and nerc!="All Regions":  d=d[d["NERC Region"]==nerc]
    if tiers and len(tiers)>0:         d=d[d["risk_category"].isin(tiers)]
    return d if len(d)>0 else DF.copy()

# ── KPI & Findings HTML ────────────────────────────────────────────
def kpi_html(df):
    try:
        n=len(df); nh=int((df["risk_category"]=="High Risk").sum())
        ns=int(df["State"].nunique())
        saidi=float(df[SC].mean()) if SC in df.columns else 0
        nat=float(DF[SC].mean()); delta=saidi-nat
        loss=float(df["estimated_annual_loss_usd"].sum()) if "estimated_annual_loss_usd" in df.columns else 0
        sla=int(df["nerc_sla_breach_risk"].sum()) if "nerc_sla_breach_risk" in df.columns else 0
        storms=int(df["weather_event_count"].sum()) if "weather_event_count" in df.columns else 0
        dc="#E74C3C" if delta>0 else "#27AE60"; da="▲" if delta>0 else "▼"
        cards=[("⚡","Utilities",f"{n:,}","analyzed","#2C3E50"),
               ("🔴","High Risk",f"{nh:,}",f"{nh/max(n,1)*100:.1f}%","#E74C3C"),
               ("🗺️","States",f"{ns}","covered","#2980B9"),
               ("📊","Avg SAIDI",f"{saidi:.0f}",f"{da}{abs(delta):.0f} vs nat.avg",dc),
               ("💰","Annual Loss",f"${loss/1e9:.2f}B",f"{sla} NERC breach","#8E44AD"),
               ("🌩️","Storm Events",f"{storms:,}","in filtered set","#16A085")]
        h='<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px;">'
        for ic,lb,vl,sb,co in cards:
            h+=(f'<div style="flex:1;min-width:100px;background:linear-gradient(135deg,{co}18,{co}35);'
                f'border:1.5px solid {co}55;border-radius:10px;padding:11px 10px;text-align:center;">'
                f'<div style="font-size:18px;">{ic}</div>'
                f'<div style="font-size:9px;color:#777;text-transform:uppercase;letter-spacing:.7px;">{lb}</div>'
                f'<div style="font-size:17px;font-weight:700;color:{co};">{vl}</div>'
                f'<div style="font-size:9px;color:#888;">{sb}</div></div>')
        return h+"</div>"
    except Exception as e:
        return f'<div style="color:#E74C3C;padding:10px;">KPI Error: {e}</div>'

def findings_html(df):
    try:
        hi=df[df["risk_category"]=="High Risk"]
        ts=hi["State"].value_counts().index[0] if len(hi)>0 else "N/A"
        tn=int(hi["State"].value_counts().iloc[0]) if len(hi)>0 else 0
        lb=float(df["estimated_annual_loss_usd"].sum())/1e9
        tn2=df.groupby("NERC Region")[SC].mean().idxmax() if SC in df.columns else "N/A"
        to=hi["Ownership"].value_counts().index[0] if len(hi)>0 else "N/A"
        items=[
            ("🗺️ Geographic Concentration",
             f"<b>{ts}</b> leads with <b>{tn}</b> High Risk utilities. "
             f"NERC region <b>{tn2}</b> has highest avg SAIDI. "
             "Southeast utilities dominate the high-risk tier.","#2980B9"),
            ("🌩️ Storm Exposure Predicts Risk",
             f"Weather features improve ROC-AUC <b>0.50→0.77</b> (+54%). "
             f"<b>{to}</b> ownership type has most High Risk utilities. "
             "Storm damage is the strongest predictor after SAIDI/SAIFI.","#E67E22"),
            ("💰 Economic & Regulatory Stakes",
             f"<b>${lb:.1f}B</b> estimated annual economic loss. "
             f"391 utilities exceed NERC SAIDI threshold of 150 min/yr. "
             "ML model: Logistic Regression, ROC-AUC 0.668, PR-AUC 0.325.","#8E44AD"),
        ]
        h='<div style="display:flex;gap:8px;flex-wrap:wrap;margin:6px 0;">'
        for title,body,col in items:
            h+=(f'<div style="flex:1;min-width:160px;background:{col}10;'
                f'border-left:4px solid {col};border-radius:0 8px 8px 0;padding:10px 12px;">'
                f'<div style="font-weight:700;color:{col};font-size:11px;margin-bottom:3px;">{title}</div>'
                f'<div style="font-size:10px;color:#bbb;line-height:1.5;">{body}</div></div>')
        return h+"</div>"
    except Exception as e:
        return f'<div style="color:#E74C3C;">Findings Error: {e}</div>'

# ── FOLIUM MAP 1 — Scatter ─────────────────────────────────────────
def folium_scatter(df):
    """Real interactive map using Folium + OpenStreetMap tiles."""
    try:
        m = folium.Map(location=[39.5,-98.35], zoom_start=4,
                       tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
                       attr="Esri, HERE, Garmin, &copy; OpenStreetMap contributors, and the GIS User Community",
                       width="100%", height="460px")
        # Build GeoJSON for all utilities
        features = []
        for _,r in df.iterrows():
            features.append({
                "type":"Feature",
                "geometry":{"type":"Point","coordinates":[float(r["lon"]),float(r["lat"])]},
                "properties":{
                    "name":str(r.get("Utility Name","")),
                    "state":str(r.get("State","")),
                    "risk":str(r.get("risk_category","")),
                    "saidi":round(float(r.get(SC,0)),1),
                    "own":str(r.get("Ownership","")),
                    "storms":int(r.get("weather_event_count",0)),
                }})
        gj = {"type":"FeatureCollection","features":features}

        def style_fn(feat):
            risk  = feat["properties"].get("risk","")
            color = RC.get(risk,"#888888")
            saidi = feat["properties"].get("saidi",0)
            rad   = max(3, min(12, saidi/80)) if saidi>0 else 4
            return {"radius":rad,"color":color,"fillColor":color,
                    "fillOpacity":0.75,"weight":1}

        folium.GeoJson(gj,
            marker=folium.CircleMarker(radius=5),
            style_function=style_fn,
            tooltip=folium.GeoJsonTooltip(
                fields=["name","state","risk","saidi","own"],
                aliases=["Utility:","State:","Risk:","SAIDI (min/yr):","Ownership:"],
                style=("background:#1a1a2e;color:#fff;font-size:12px;"
                       "border:1px solid #444;border-radius:6px;padding:6px;")),
            popup=folium.GeoJsonPopup(
                fields=["name","state","risk","saidi","storms","own"],
                aliases=["<b>Utility</b>","State","Risk","SAIDI (min/yr)","Storm Events","Ownership"])
        ).add_to(m)

        m.get_root().html.add_child(folium.Element("""
        <div style="position:fixed;bottom:25px;left:25px;z-index:9999;
            background:rgba(14,17,23,0.92);padding:10px 14px;border-radius:8px;
            border:1px solid #333;font-family:sans-serif;font-size:12px;color:#eee;">
          <div style="font-weight:700;margin-bottom:5px;">⚡ Risk Tier</div>
          <div><span style="color:#E74C3C;font-size:15px;">●</span> High Risk</div>
          <div><span style="color:#F39C12;font-size:15px;">●</span> Medium Risk</div>
          <div><span style="color:#27AE60;font-size:15px;">●</span> Low Risk</div>
          <div style="color:#888;font-size:9px;margin-top:4px;">Dot size = SAIDI severity<br>Click dot for details</div>
        </div>"""))
        return m._repr_html_()
    except Exception as e:
        return f'<div style="color:#E74C3C;padding:20px;">Scatter map error: {e}</div>'

# ── FOLIUM MAP 2 — KMeans Clusters ────────────────────────────────
def folium_clusters(df, k=6):
    """KMeans clusters on real interactive Folium map."""
    try:
        k = max(2, min(int(k), len(df)-1))
        d = df.copy()
        SC2l = SC2 if SC2 in d.columns else SC
        fcols = [c for c in [SC,SC2l,"weather_event_count","total_damage_usd","lat","lon"]
                 if c in d.columns]
        X = StandardScaler().fit_transform(d[fcols].fillna(0))
        d["cl"] = KMeans(n_clusters=k,random_state=42,n_init=3).fit_predict(X)

        stats = d.groupby("cl").agg(n=("cl","count"),phr=("high_risk","mean"),
                                     avgs=(SC,"mean")).reset_index()
        lmap = {int(r["cl"]):
                f"Cluster {int(r['cl'])}: {int(r['n'])} utils · "
                f"{r['phr']*100:.0f}% HR · SAIDI {r['avgs']:.0f} min/yr"
                for _,r in stats.iterrows()}

        m = folium.Map(location=[39.5,-98.35], zoom_start=4,
                       tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
                       attr="Esri, HERE, Garmin, &copy; OpenStreetMap contributors, and the GIS User Community",
                       width="100%", height="460px")

        # One FeatureGroup per cluster for layer control
        for cid in range(k):
            c_df = d[d["cl"]==cid]
            if len(c_df)==0: continue
            color = HP[cid % len(HP)]
            label = lmap.get(cid, f"Cluster {cid}")
            fg = folium.FeatureGroup(name=f'<span style="color:{color}">■</span> {label}')
            for _,r in c_df.iterrows():
                folium.CircleMarker(
                    location=[float(r["lat"]),float(r["lon"])],
                    radius=6, color=color, fill=True, fill_color=color,
                    fill_opacity=0.8, weight=1,
                    tooltip=(f"<b>{r.get('Utility Name','')}</b><br>"
                             f"State: {r.get('State','')} | "
                             f"Risk: {r.get('risk_category','')} | "
                             f"SAIDI: {r.get(SC,0):.0f}<br>"
                             f"{label}"),
                ).add_to(fg)
            # Centroid marker
            folium.Marker(
                location=[float(c_df["lat"].mean()), float(c_df["lon"].mean())],
                icon=folium.DivIcon(html=(
                    f'<div style="background:{color};color:#fff;font-weight:700;'
                    f'font-size:11px;padding:3px 7px;border-radius:50%;'
                    f'border:2px solid #fff;opacity:0.9;">C{cid}</div>')),
                tooltip=label
            ).add_to(fg)
            fg.add_to(m)

        folium.LayerControl(collapsed=False,position="topright").add_to(m)
        return m._repr_html_()
    except Exception as e:
        return f'<div style="color:#E74C3C;padding:20px;">Cluster map error: {e}</div>'

# ── Choropleth (plotly — works fine) ──────────────────────────────
def choropleth(df, metric):
    try:
        if metric=="Average SAIDI (min/yr)":
            sd=df.groupby("State")[SC].mean().reset_index(); sd.columns=["State","value"]
            cl,cs,t="Avg SAIDI","RdYlGn_r","Average Outage Duration by State"
        elif metric=="High Risk Utility %":
            tmp=df.copy(); tmp["v"]=(tmp["risk_category"]=="High Risk").astype(float)*100
            sd=tmp.groupby("State")["v"].mean().reset_index(); sd.columns=["State","value"]
            cl,cs,t="% High Risk","Reds","% High Risk Utilities by State"
        elif metric=="Total Storm Damage ($B)":
            sd=df.groupby("State")["total_damage_usd"].sum().div(1e9).reset_index(); sd.columns=["State","value"]
            cl,cs,t="Damage ($B)","OrRd","Total Storm Damage by State"
        else:
            sd=df.groupby("State")["estimated_annual_loss_usd"].sum().div(1e6).reset_index(); sd.columns=["State","value"]
            cl,cs,t="Loss ($M)","Purples","Estimated Annual Economic Loss by State ($M)"
        fig=px.choropleth(sd,locations="State",locationmode="USA-states",
            color="value",color_continuous_scale=cs,scope="usa",title=t,labels={"value":cl})
        fig.update_layout(title_x=0.5,margin=dict(l=0,r=0,t=50,b=0),
            paper_bgcolor=BG,font_color=TX,
            geo=dict(bgcolor=BG,landcolor="#1E2530",lakecolor=BG,
                     showlakes=True,showland=True,showcoastlines=True,coastlinecolor="#333"))
        return fig
    except Exception as e: return ef(f"Choropleth: {e}")

# ── Analysis Charts ────────────────────────────────────────────────
def risk_dist(df):
    try:
        fig=make_subplots(rows=1,cols=2,specs=[[{"type":"pie"},{"type":"bar"}]],
            subplot_titles=["Risk Tier Distribution","Risk by Ownership Type"])
        rc=df["risk_category"].value_counts()
        fig.add_trace(go.Pie(labels=rc.index,values=rc.values,hole=.55,
            marker_colors=[RC.get(r,"#999") for r in rc.index],name=""),row=1,col=1)
        if "Ownership" in df.columns:
            major=df["Ownership"].value_counts().head(6).index.tolist()
            g=df[df["Ownership"].isin(major)].groupby(["Ownership","risk_category"]).size().reset_index(name="Count")
            for tier in TIERS:
                t=g[g["risk_category"]==tier]
                if len(t)>0:
                    fig.add_trace(go.Bar(x=t["Ownership"],y=t["Count"],name=tier,
                        marker_color=RC.get(tier,"#999")),row=1,col=2)
        fig.update_layout(barmode="stack",paper_bgcolor=BG,plot_bgcolor=BG,font_color=TX,height=400,
            legend=dict(bgcolor="#1E2530",font=dict(color=TX)),margin=dict(l=20,r=20,t=60,b=70))
        fig.update_xaxes(tickangle=35,tickfont_size=8,gridcolor=GR)
        fig.update_yaxes(gridcolor=GR)
        return fig
    except Exception as e: return ef(f"Risk dist: {e}")

def saidi_violin(df):
    try:
        fig=go.Figure(); q95=df[SC].quantile(.95)
        for tier in TIERS:
            t=df[(df["risk_category"]==tier)&(df[SC]>0)&(df[SC]<q95)]
            if len(t)>0:
                fig.add_trace(go.Violin(y=t[SC],name=tier,box_visible=True,meanline_visible=True,
                    fillcolor=RC.get(tier,"#999"),opacity=.75,line_color=RC.get(tier,"#999")))
        return lyt(fig,400,"SAIDI Distribution by Risk Tier (non-zero utilities)")
    except Exception as e: return ef(f"Violin: {e}")

def risk_hist(df):
    try:
        thr=df["risk_score"].quantile(.80); fig=go.Figure()
        for tier in TIERS:
            t=df[df["risk_category"]==tier]["risk_score"]
            if len(t)>0:
                fig.add_trace(go.Histogram(x=t,name=tier,nbinsx=40,
                    marker_color=RC.get(tier,"#999"),opacity=.8))
        fig.add_vline(x=thr,line_dash="dash",line_color="#FFD700",line_width=2)
        fig.add_annotation(x=thr,y=1,yref="paper",text=f"Top 20%\n{thr:.3f}",
            showarrow=True,arrowhead=2,arrowcolor="#FFD700",
            font=dict(color="#FFD700",size=10),bgcolor="#1E2530",ax=40,ay=-30)
        fig=lyt(fig,380,"Risk Score Distribution — Composite SAIDI/SAIFI Percentile Rank")
        fig.update_layout(barmode="overlay")
        return fig
    except Exception as e: return ef(f"Histogram: {e}")

def nerc_heatmap(df):
    try:
        major=df["NERC Region"].value_counts().head(10).index.tolist()
        sub=df[df["NERC Region"].isin(major)]
        ns=sub.groupby("NERC Region").agg(avg_saidi=(SC,"mean"),pct_high=("high_risk","mean"),
            avg_storms=("weather_event_count","mean"),avg_loss=("estimated_annual_loss_usd","mean")).reset_index()
        ns=ns.sort_values("avg_saidi",ascending=False)
        metrics=["avg_saidi","pct_high","avg_storms","avg_loss"]
        labels=["Avg SAIDI\n(min/yr)","% High Risk","Avg Storm\nEvents","Avg Loss\n($M)"]
        z,txt=[],[]
        for m,l in zip(metrics,labels):
            col=ns[m].values.astype(float)
            if m=="avg_loss": col=col/1e6
            mn,mx=col.min(),col.max()
            z.append(((col-mn)/(mx-mn+1e-9)).tolist())
            fmt=".0f" if "saidi" in m or "storm" in m else ".1f" if "loss" in m else ".1%"
            txt.append([f"{v:{fmt}}" for v in col])
        fig=go.Figure(go.Heatmap(z=z,x=ns["NERC Region"].tolist(),y=labels,
            text=txt,texttemplate="%{text}",colorscale="RdYlGn_r",showscale=False,xgap=2,ygap=2))
        fig=lyt(fig,280,"NERC Region Risk Heatmap")
        fig.update_xaxes(tickangle=30,tickfont_size=9)
        return fig
    except Exception as e: return ef(f"NERC heatmap: {e}")

def radar_chart(df):
    try:
        major=df["NERC Region"].value_counts().head(7).index.tolist()
        sub=df[df["NERC Region"].isin(major)]
        dims={"Avg SAIDI":(SC,"mean"),"% High Risk":("high_risk","mean"),
              "Storm Events":("weather_event_count","mean"),
              "Damage ($M)":("total_damage_usd","mean"),
              "Human Impact":("human_impact_score","mean")}
        stats=sub.groupby("NERC Region").agg(**{k:(v[0],v[1]) for k,v in dims.items()}).reset_index()
        for col in dims.keys():
            mn,mx=stats[col].min(),stats[col].max()
            stats[col]=(stats[col]-mn)/(mx-mn+1e-9)
        cats=list(dims.keys())
        fig=go.Figure()
        for i,(_,row) in enumerate(stats.iterrows()):
            vals=[row[c] for c in cats]+[row[cats[0]]]
            fig.add_trace(go.Scatterpolar(r=vals,theta=cats+[cats[0]],
                name=row["NERC Region"],fill="toself",opacity=.6,
                line=dict(color=HP[i%len(HP)],width=1.5)))
        fig.update_layout(
            polar=dict(bgcolor="#151B23",
                radialaxis=dict(visible=True,range=[0,1],tickfont=dict(size=8,color="#888"),gridcolor=GR),
                angularaxis=dict(tickfont=dict(size=9,color=TX),gridcolor=GR)),
            paper_bgcolor=BG,font_color=TX,height=420,
            legend=dict(bgcolor="#1E2530",font=dict(color=TX,size=9)),
            title=dict(text="NERC Region Risk Radar — 5 Dimensions",x=0.5,font=dict(size=12)),
            margin=dict(l=60,r=60,t=60,b=20))
        return fig
    except Exception as e: return ef(f"Radar: {e}")

def bubble(df):
    try:
        d=df[(df[SC]>0)&(df["weather_event_count"]>0)&(df["estimated_annual_loss_usd"]>0)].copy()
        if len(d)==0: return ef("No utilities with non-zero SAIDI + storm events")
        d["lm"]=d["estimated_annual_loss_usd"]/1e6
        d["sz"]=((d["lm"]/d["lm"].max()*40+5)).clip(4,45)
        fig=go.Figure()
        for tier in TIERS:
            t=d[d["risk_category"]==tier]
            if len(t)==0: continue
            fig.add_trace(go.Scatter(x=t["weather_event_count"],y=t[SC],mode="markers",name=tier,
                marker=dict(color=RC.get(tier,"#999"),size=t["sz"],opacity=.65,
                            line=dict(width=.4,color="#ffffff30")),
                text=("<b>"+t["Utility Name"].fillna("")+"</b><br>"+
                      "State: "+t["State"].fillna("")+"<br>"+
                      "SAIDI: "+t[SC].round(1).astype(str)+" min/yr<br>"+
                      "Loss: $"+t["lm"].round(1).astype(str)+"M"),
                hovertemplate="%{text}<extra></extra>"))
        fig=lyt(fig,420,"Storm Events vs SAIDI — bubble size = economic loss")
        fig.update_xaxes(title_text="Storm Event Count")
        fig.update_yaxes(title_text="SAIDI (min/yr)")
        return fig
    except Exception as e: return ef(f"Bubble: {e}")

def state_bar(df):
    try:
        sc2=df[df["risk_category"]=="High Risk"]["State"].value_counts().head(20).reset_index()
        sc2.columns=["State","Count"]
        if len(sc2)==0: return ef("No High Risk utilities")
        fig=px.bar(sc2,x="State",y="Count",color="Count",color_continuous_scale="Reds",
            title="Top 20 States by High Risk Utility Count",text="Count")
        fig.update_traces(textposition="outside",textfont_size=8)
        fig=lyt(fig,400); fig.update_xaxes(tickangle=45)
        fig.update_coloraxes(showscale=False); return fig
    except Exception as e: return ef(f"State bar: {e}")

def econ_bar(df):
    try:
        top=(df[df["estimated_annual_loss_usd"]>0][["Utility Name","State","risk_category","estimated_annual_loss_usd"]]
             .sort_values("estimated_annual_loss_usd",ascending=False).head(20).copy())
        if len(top)==0: return ef("No economic loss data")
        top["Loss ($M)"]=(top["estimated_annual_loss_usd"]/1e6).round(1)
        top["Label"]=top["Utility Name"].str[:24]+" ("+top["State"]+")"
        fig=px.bar(top,x="Loss ($M)",y="Label",orientation="h",
            color="risk_category",color_discrete_map=RC,
            title="Top 20 Utilities — Annual Economic Loss ($M)",text="Loss ($M)")
        fig.update_traces(textposition="outside",textfont_size=8)
        fig=lyt(fig,520); fig.update_yaxes(categoryorder="total ascending")
        return fig
    except Exception as e: return ef(f"Economic: {e}")

def comparison(u1,u2):
    try:
        if not u1 or not u2: return ef("Select two utilities to compare")
        r1=DF[DF["Utility Name"]==u1]; r2=DF[DF["Utility Name"]==u2]
        if len(r1)==0 or len(r2)==0: return ef("Utility not found")
        r1,r2=r1.iloc[0],r2.iloc[0]
        dims={"SAIDI (min/yr)":(SC,1),"SAIFI (x/yr)":(SC2,1),
              "Risk Score":("risk_score",1),"Storm Events":("weather_event_count",1),
              "Damage ($M)":("total_damage_usd",1e6),"Human Impact":("human_impact_score",1)}
        labs,v1,v2=[],[],[]
        for lab,(col,div) in dims.items():
            labs.append(lab)
            v1.append(round(float(r1.get(col,0) if col in r1.index else 0)/div,3))
            v2.append(round(float(r2.get(col,0) if col in r2.index else 0)/div,3))
        c1=RC.get(str(r1.get("risk_category","")),"#2980B9")
        c2=RC.get(str(r2.get("risk_category","")),"#16A085")
        fig=go.Figure()
        fig.add_trace(go.Bar(name=f"{u1[:25]} [{r1.get('risk_category','')}]",
            x=labs,y=v1,marker_color=c1,opacity=.85,
            text=[str(v) for v in v1],textposition="outside",textfont_size=8))
        fig.add_trace(go.Bar(name=f"{u2[:25]} [{r2.get('risk_category','')}]",
            x=labs,y=v2,marker_color=c2,opacity=.85,
            text=[str(v) for v in v2],textposition="outside",textfont_size=8))
        fig=lyt(fig,400,f"{u1[:22]} vs {u2[:22]}")
        fig.update_layout(barmode="group"); return fig
    except Exception as e: return ef(f"Comparison: {e}")

def tbl(df):
    want=["Utility Name","State","Ownership","NERC Region","risk_category","risk_score",
          SC,SC2,"weather_event_count","estimated_annual_loss_usd","nerc_sla_breach_risk"]
    cols=[c for c in want if c in df.columns]
    out=df[cols].copy().rename(columns={"risk_category":"Risk Tier","risk_score":"Risk Score",
        SC:"SAIDI (min/yr)",SC2:"SAIFI (x/yr)","weather_event_count":"Storm Events",
        "estimated_annual_loss_usd":"Est. Loss ($M)","nerc_sla_breach_risk":"NERC Breach"})
    if "Risk Score" in out.columns:
        out["Risk Score"]=out["Risk Score"].round(4)
        out=out.sort_values("Risk Score",ascending=False)
    if "SAIDI (min/yr)" in out.columns: out["SAIDI (min/yr)"]=out["SAIDI (min/yr)"].round(1)
    if "Est. Loss ($M)" in out.columns: out["Est. Loss ($M)"]=(out["Est. Loss ($M)"]/1e6).round(1)
    return out.reset_index(drop=True)

# ── Master update ──────────────────────────────────────────────────
def update(state,own,nerc,tiers,metric,k):
    try:
        d=filt(state,own,nerc,tiers)
        return (kpi_html(d), findings_html(d),
                choropleth(d,metric),
                folium_scatter(d),
                folium_clusters(d,k),
                risk_dist(d), saidi_violin(d), risk_hist(d),
                nerc_heatmap(d), radar_chart(d), bubble(d),
                state_bar(d), econ_bar(d), tbl(d))
    except Exception as e:
        eh=f'<div style="color:#E74C3C;padding:10px;">Error: {e}</div>'
        emp=ef(str(e))
        return (eh,eh,emp,_map_html(),_map_html(),emp,emp,emp,emp,emp,emp,emp,emp,pd.DataFrame())

def startup():
    d=DF.copy()
    skip=ef("Change any filter to load this chart")
    return (kpi_html(d), findings_html(d),
            choropleth(d,"Average SAIDI (min/yr)"),
            _map_html(), _map_html(),
            risk_dist(d), skip, skip, skip, skip, skip,
            state_bar(d), econ_bar(d), tbl(d))

def cmp_update(u1,u2): return comparison(u1,u2)

# ── UI ─────────────────────────────────────────────────────────────
def build():
    all_st =["All States"] +sorted(DF["State"].dropna().unique().tolist())
    all_own=["All Types"]  +sorted(DF["Ownership"].dropna().unique().tolist())
    all_nr =["All Regions"]+sorted(DF["NERC Region"].dropna().unique().tolist())

    with gr.Blocks(
        title="⚡ Power Outage Risk Dashboard",
        css=".gradio-container{background:#0E1117!important}body{background:#0E1117!important}footer{display:none!important}"
    ) as demo:

        gr.HTML("""
        <div style="background:linear-gradient(135deg,#1a1f2e,#2C3E50);border-radius:12px;
                    padding:20px 26px;margin-bottom:12px;border:1px solid #2A3040;">
          <h1 style="font-size:22px;font-weight:800;color:#FAFAFA;margin:0 0 4px 0;">
            ⚡ Power Outage Risk Dashboard
          </h1>
          <p style="font-size:11px;color:#888;margin:0;line-height:1.6;">
            EIA-861 + NOAA Storm Events 2024 · 1,677 utilities · 50 states · 3.4M records ·
            Logistic Regression + Weather features · ROC-AUC 0.668 · PR-AUC 0.325 ·
            Built by <strong style="color:#aaa">Sejal Khade</strong> — MS Data Science, UT Arlington
          </p>
        </div>""")

        with gr.Row():
            with gr.Column(scale=1,min_width=200):
                gr.HTML('<p style="font-size:10px;color:#888;text-transform:uppercase;letter-spacing:.8px;font-weight:600;margin:0 0 6px;">🔍 Filters</p>')
                st_dd =gr.Dropdown(choices=all_st, value="All States",  label="State",         interactive=True)
                own_dd=gr.Dropdown(choices=all_own,value="All Types",   label="Ownership Type",interactive=True)
                nr_dd =gr.Dropdown(choices=all_nr, value="All Regions", label="NERC Region",   interactive=True)
                tier_cb=gr.CheckboxGroup(choices=TIERS,value=TIERS,label="Risk Tier",interactive=True)
                gr.HTML('<p style="font-size:10px;color:#888;text-transform:uppercase;letter-spacing:.8px;font-weight:600;margin:12px 0 6px;">🗺️ Map Controls</p>')
                met_dd=gr.Dropdown(
                    choices=["Average SAIDI (min/yr)","High Risk Utility %",
                             "Total Storm Damage ($B)","Economic Loss ($M)"],
                    value="Average SAIDI (min/yr)",label="Choropleth Metric",interactive=True)
                k_sl=gr.Slider(minimum=3,maximum=10,step=1,value=6,label="K — Clusters",interactive=True)

            with gr.Column(scale=4):
                kpi_out=gr.HTML(); find_out=gr.HTML()

                gr.HTML('<div style="font-size:13px;font-weight:700;color:#FAFAFA;padding:10px 0 6px;border-bottom:1px solid #2A3040;margin-bottom:8px;">🗺️ Geographic Analysis</div>')
                with gr.Tabs():
                    with gr.Tab("🌍 Choropleth — State Level"):
                        choro_out=gr.Plot(show_label=False)
                    with gr.Tab("📍 Utility Scatter Map"):
                        scat_out=gr.HTML(value=_map_html())
                    with gr.Tab("🔵 KMeans Cluster Map"):
                        clust_out=gr.HTML(value=_map_html())

                gr.HTML('<div style="font-size:13px;font-weight:700;color:#FAFAFA;padding:10px 0 6px;border-bottom:1px solid #2A3040;margin-bottom:8px;">📊 Risk Analysis</div>')
                with gr.Tabs():
                    with gr.Tab("Risk Distribution"): rdist_out=gr.Plot(show_label=False)
                    with gr.Tab("SAIDI Violin"):      viol_out=gr.Plot(show_label=False)
                    with gr.Tab("Risk Histogram"):    hist_out=gr.Plot(show_label=False)
                    with gr.Tab("NERC Heatmap"):      nerc_out=gr.Plot(show_label=False)
                    with gr.Tab("NERC Radar"):        radar_out=gr.Plot(show_label=False)
                    with gr.Tab("Bubble Chart"):      bub_out=gr.Plot(show_label=False)
                    with gr.Tab("State Rankings"):    sbar_out=gr.Plot(show_label=False)
                    with gr.Tab("Economic Impact"):   econ_out=gr.Plot(show_label=False)

                gr.HTML('<div style="font-size:13px;font-weight:700;color:#FAFAFA;padding:10px 0 6px;border-bottom:1px solid #2A3040;margin-bottom:8px;">🔍 Utility Comparison Tool</div>')
                with gr.Row():
                    u1_dd=gr.Dropdown(choices=UNAMES,value=UNAMES[0] if UNAMES else None,
                                      label="Utility A — type to search",interactive=True,filterable=True)
                    u2_dd=gr.Dropdown(choices=UNAMES,value=UNAMES[1] if len(UNAMES)>1 else None,
                                      label="Utility B — type to search",interactive=True,filterable=True)
                comp_out=gr.Plot(show_label=False)

                gr.HTML('<div style="font-size:13px;font-weight:700;color:#FAFAFA;padding:10px 0 6px;border-bottom:1px solid #2A3040;margin-bottom:8px;">📋 Utility Data</div>')
                tbl_out=gr.Dataframe(interactive=False,wrap=False)
                dl_btn=gr.Button("⬇️ Download Filtered CSV",variant="secondary",size="sm")
                dl_file=gr.File(label="Download",visible=False)

        gr.HTML("""
        <div style="text-align:center;padding:12px 0 4px;color:#444;font-size:10px;border-top:1px solid #2A3040;margin-top:16px;">
          <strong style="color:#666">Sejal Khade</strong> · MS Data Science, UT Arlington ·
          <a href="https://github.com/SejalKhade" style="color:#E74C3C;">github.com/SejalKhade</a>
          · EIA-861 + NOAA Storm Events 2024
        </div>""")

        inputs=[st_dd,own_dd,nr_dd,tier_cb,met_dd,k_sl]
        outputs=[kpi_out,find_out,choro_out,scat_out,clust_out,
                 rdist_out,viol_out,hist_out,nerc_out,radar_out,bub_out,
                 sbar_out,econ_out,tbl_out]

        for w in inputs:
            w.change(fn=update,inputs=inputs,outputs=outputs)
        for w in [u1_dd,u2_dd]:
            w.change(fn=cmp_update,inputs=[u1_dd,u2_dd],outputs=comp_out)

        def do_dl(state,own,nerc,tiers,*_):
            d=filt(state,own,nerc,tiers); p="/tmp/outage_risk.csv"
            tbl(d).to_csv(p,index=False); return gr.File(value=p,visible=True)

        dl_btn.click(fn=do_dl,inputs=inputs,outputs=dl_file)
        demo.load(fn=startup,inputs=None,outputs=outputs)
    return demo

if __name__ == "__main__":
    app = build()
    app.launch(server_name="0.0.0.0",server_port=7860,
               share=False,ssr_mode=False,show_error=True)