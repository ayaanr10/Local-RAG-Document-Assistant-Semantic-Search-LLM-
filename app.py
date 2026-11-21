"""Atlas RAG — clean document-question-answering dashboard.

Run locally: streamlit run app.py
Render: streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
"""
from __future__ import annotations

import csv
from datetime import datetime
import io
import os
from pathlib import Path
import time
import uuid

import streamlit as st
from dotenv import load_dotenv

from rag.answering import answer_question
from rag.citations import format_answer_citations, render_reference_key
from rag.engine import RAGEngine
from rag.evaluation import evaluate_retrieval

load_dotenv()
st.set_page_config(page_title='Atlas RAG | Document Intelligence', page_icon='🧭', layout='wide',
                   initial_sidebar_state='expanded')

PUBLIC_DEMO = os.getenv('PUBLIC_DEMO', '0') == '1'
ROOT = Path(__file__).parent

st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"],.stApp{font-family:'DM Sans',sans-serif;color:#E5EDF6}
.stApp{background:radial-gradient(circle at 86% 0%,#14395c 0%,transparent 27%),#081322}
[data-testid="stSidebar"]{background:#0c1b2d;border-right:1px solid #23415b}
[data-testid="stHeader"]{background:transparent}
#MainMenu,footer{visibility:hidden}
.block-container{max-width:1260px;padding-top:2.1rem;padding-bottom:4rem}
.headline{font-family:'Manrope',sans-serif;font-size:2.35rem;font-weight:800;letter-spacing:-.05em;line-height:1.14;margin:0 0 .2rem;color:#f4f8fc}
.kicker{color:#67e2d0;font-size:.76rem;font-weight:800;letter-spacing:.16em;text-transform:uppercase;margin-bottom:.6rem}
.subtitle{color:#a8bdcf;font-size:1.0rem;margin-bottom:1rem}
.panel{background:#112338;border:1px solid #29415c;border-radius:17px;padding:1.08rem 1.18rem;margin-bottom:.7rem}
.metric-title{color:#9fb4c9;font-size:.81rem;font-weight:600}
.metric-value{color:#f0f6ff;font-size:1.72rem;font-family:Manrope,sans-serif;font-weight:800;margin-top:.23rem;letter-spacing:-.04em}
.metric-foot{color:#66dfc9;font-size:.72rem;margin-top:.26rem}
.brand{font-family:Manrope,sans-serif;font-size:1.16rem;font-weight:800;color:#eaf8ff}
.pill{display:inline-block;background:#163f43;border:1px solid #2d736e;color:#81f1de;padding:4px 11px;border-radius:22px;font-weight:700;font-size:11px}
.tag{color:#8daac7;font-size:.83rem}
[data-testid="stMetric"]{background:#112338;border:1px solid #29415c;border-radius:16px;padding:1rem}
.stButton button[kind="primary"]{background:#1fc7b0;color:#04202b;border:0;font-weight:800;border-radius:12px}
.stButton button[kind="secondary"]{border-color:#45627b;border-radius:12px}
[data-testid="stChatMessage"]{background:#102238;border:1px solid #25415c;border-radius:15px;margin-bottom:.7rem}
[data-testid="stFileUploader"]{background:#10263a;border:1px dashed #4a6a80;border-radius:13px;padding:.4rem}
[data-testid="stExpander"]{background:#102238;border:1px solid #27435c;border-radius:13px}
/* Small teal source markers instead of repeating long filenames inline. */
.atlas-citation{display:inline-block;vertical-align:super;margin-left:3px;padding:0px 5px;
  border:1px solid #427e82;border-radius:7px;background:#183a43;color:#80e5d5;
  font-size:.67em;font-weight:700;line-height:1.55;white-space:nowrap}
.atlas-source-key{margin-top:.65rem;margin-bottom:.45rem;padding:.7rem .9rem;
  border-top:1px solid #2a4356;color:#9fb8c9;font-size:.75rem;line-height:1.7}
.atlas-source-heading{font-size:.7rem;font-weight:700;text-transform:uppercase;
  letter-spacing:.06em;color:#90b1c4;margin-bottom:.25rem}
.atlas-source-item{display:block;overflow-wrap:anywhere;margin:.1rem 0}
.atlas-source-item b{color:#80e5d5;margin-right:5px}
.atlas-source-page{color:#8fa5b9}
[data-baseweb="tab-list"]{border-bottom:1px solid #284159;gap:.6rem}
[data-baseweb="tab"]{font-weight:700}
hr{border-color:#233c51}
</style>''', unsafe_allow_html=True)

if 'session_id' not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:20]
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'telemetry' not in st.session_state:
    st.session_state.telemetry = []
if 'pending' not in st.session_state:
    st.session_state.pending = None


@st.cache_resource(show_spinner='Opening knowledge base…')
def get_engine(namespace: str) -> RAGEngine:
    path = os.getenv('RAG_DB_PATH', 'data/chroma')
    return RAGEngine(path, collection_name=namespace)


namespace = f"session_{st.session_state.session_id}" if PUBLIC_DEMO else 'rag_docs'
try:
    engine = get_engine(namespace)
except Exception as exc:
    st.error('Could not initialize ChromaDB. Check server logs and installed requirements.')
    if not PUBLIC_DEMO:
        st.exception(exc)
    st.stop()


def show_answer(text: str) -> None:
    """Display readable answer text with compact, deduplicated source markers."""
    body, sources = format_answer_citations(text)
    st.markdown(body, unsafe_allow_html=True)
    if sources:
        st.markdown(render_reference_key(sources), unsafe_allow_html=True)


def show_sources(matches: list[dict]) -> None:
    if not matches:
        st.info('No matching passages passed the selected relevance filter.')
        return
    for rank, match in enumerate(matches, 1):
        # Avoid nested expanders when this is shown inside a chat response's source panel.
        with st.container(border=True):
            st.markdown(f"**Source {rank}: {match['source']} · page/section {match['page']}**")
            st.text(match['text'])
            st.caption(f"Vector distance: {match['distance']:.3f} (lower means more similar, not necessarily correct)")


def demo_load() -> None:
    for path in sorted((ROOT / 'samples').glob('*')):
        if path.suffix.lower() in {'.pdf', '.md', '.txt'}:
            engine.add_document(path.name, path.read_bytes())


def metric(title: str, value: str, detail: str) -> None:
    st.markdown(f'<div class="panel"><div class="metric-title">{title}</div>'
                f'<div class="metric-value">{value}</div><div class="metric-foot">{detail}</div></div>',
                unsafe_allow_html=True)


with st.sidebar:
    st.markdown('<div class="brand">◈ &nbsp;ATLAS / RAG</div>', unsafe_allow_html=True)
    st.caption('Knowledge engineering studio')
    st.divider()
    st.subheader('📁 Knowledge sources')
    files = st.file_uploader('PDF, Markdown or plain text', type=['pdf','md','txt'],
                             accept_multiple_files=True, help='PDF limit: 20 MB; TXT/MD: 5 MB each')
    if st.button('⬆ Index selected files', type='primary', use_container_width=True, disabled=not files):
        for file in files or []:
            try:
                if PUBLIC_DEMO and engine.stats()['documents'] >= 5:
                    st.warning('Public demo limit: at most five indexed documents per browser session.')
                    break
                with st.spinner(f'Indexing {file.name}…'):
                    result = engine.add_document(file.name, file.getvalue())
                st.success(f"{result['name']}: {result['status']} · {result['chunks']} chunks")
            except Exception as exc:
                st.error(f'{file.name}: {exc}')
    with st.expander('Demo tools', expanded=False):
        if st.button('Load sample documents', use_container_width=True):
            try:
                with st.spinner('Indexing sample documents…'):
                    demo_load()
                st.success('Sample documents added to your knowledge base.')
                st.rerun()
            except Exception as exc:
                st.error(f'Unable to index samples: {exc}')

    st.divider()
    with st.expander('Advanced retrieval settings', expanded=False):
        st.caption('Optional controls for retrieval experiments.')
        top_k = st.slider('Top-K passages', min_value=1, max_value=8, value=4)
        context_budget = st.select_slider('Context budget',
                 options=[2000,4000,6000,8000,12000,16000], value=8000,
                 format_func=lambda x: f'{x:,} characters')
        strict_filter = st.toggle('Filter distant results', value=False,
                                  help='Experimental cosine-distance filter; adjust after testing.')
        max_distance = st.slider('Maximum cosine distance', .2, 1.8, 1.15, .05,
                                 disabled=not strict_filter)
    if PUBLIC_DEMO:
        st.caption('Public demo: do not upload confidential files. Indexed documents may be lost after a server restart.')
    else:
        st.caption('Local document index. If Gemini is enabled, retrieved excerpts are sent to Google.')

st.markdown('<div class="kicker">Document intelligence / retrieval-augmented generation</div>', unsafe_allow_html=True)
st.markdown('<div class="headline">Ask better questions.<br>Get evidence-backed answers.</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a document, ask a question, and get answers with sources.</div>', unsafe_allow_html=True)

stats = engine.stats()
last = st.session_state.telemetry[-1] if st.session_state.telemetry else None
c1,c2,c3 = st.columns(3)
with c1: metric('DOCUMENTS', str(stats['documents']), 'Indexed knowledge sources')
with c2: metric('SEARCHABLE CHUNKS', str(stats['chunks']), '384-dimensional embeddings')
with c3: metric('QUESTIONS ASKED', str(len(st.session_state.telemetry)), 'This browser session')

chat_tab,library_tab,eval_tab = st.tabs(['💬  Ask documents','📚  Documents','📊  Evaluation'])

with chat_tab:
    st.markdown('### Ask your documents')
    if not stats['documents']:
        st.info('Upload a PDF, Markdown or text file in the sidebar, then click **Index selected files**. Or open Demo tools to load sample documents.')

    for item in st.session_state.messages:
        with st.chat_message(item['role']):
            if item['role'] == 'assistant':
                show_answer(item['text'])
            else:
                st.markdown(item['text'])
            if item.get('matches'):
                with st.expander('View supporting passages'):
                    show_sources(item['matches'])

    demo_question_limit = PUBLIC_DEMO and len(st.session_state.telemetry) >= 12
    if demo_question_limit:
        st.info('Public demo question limit reached for this browser session (12 questions).')
    question = st.chat_input('Ask a question about your documents…',
                             disabled=not stats['documents'] or demo_question_limit)
    if question and stats['documents']:
        st.session_state.messages.append({'role':'user','text':question})
        with st.chat_message('user'):
            st.markdown(question)
        with st.chat_message('assistant'):
            try:
                with st.spinner('Searching your documents…'):
                    t0=time.perf_counter()
                    matches=engine.search(question,top_k=top_k,
                            max_distance=max_distance if strict_filter else None)
                    retrieval_ms=(time.perf_counter()-t0)*1000
                with st.spinner('Preparing an evidence-backed answer…'):
                    t1=time.perf_counter()
                    result=answer_question(question,matches,max_context_chars=context_budget)
                    generation_ms=(time.perf_counter()-t1)*1000
                show_answer(result.text)
                with st.expander('View supporting passages', expanded=result.mode!='gemini'):
                    show_sources(matches)
                st.session_state.messages.append({'role':'assistant','text':result.text,
                    'matches':matches})
                st.session_state.telemetry.append({
                    'time':datetime.now().strftime('%H:%M:%S'), 'question':question,
                    'retrieval_ms':retrieval_ms,'generation_ms':generation_ms,
                    'est_tokens':result.estimated_input_tokens,
                    'actual_input_tokens':result.actual_input_tokens,
                    'actual_output_tokens':result.actual_output_tokens,
                    'sources':result.used_sources,'mode':result.mode})
            except Exception as exc:
                st.error(f'Question failed: {exc}')
        st.rerun()

    with st.expander('Retrieval details', expanded=False):
        if last:
            st.markdown(f"**Last question:** {last['question']}")
            a,b,c=st.columns(3)
            a.metric('Retrieval time', f"{last['retrieval_ms']:.0f} ms")
            st.caption(f"Gemini generation time: {last.get('generation_ms', 0) / 1000:.1f} seconds")
            b.metric('Estimated input tokens', f"{last['est_tokens']:,}")
            c.metric('Sources used', str(last['sources']))
            st.caption('Token estimate uses an approximate characters/4 rule; actual API counts can differ.')
            if last['actual_input_tokens'] is not None or last['actual_output_tokens'] is not None:
                left,right=st.columns(2)
                left.metric('Reported API input tokens', last['actual_input_tokens'] if last['actual_input_tokens'] is not None else '—')
                right.metric('Reported API output tokens', last['actual_output_tokens'] if last['actual_output_tokens'] is not None else '—')
        else:
            st.caption('Ask a question to see search time, retrieved passages and token usage.')

with library_tab:
    st.markdown('### Indexed document library')
    st.caption('Manage your knowledge base. Page metadata is retained for source inspection.')
    docs=engine.list_documents()
    if not docs:
        st.info('Your collection is empty. Upload a PDF/TXT/Markdown file or load sample docs.')
    else:
        for doc in docs:
            a,b,c,d=st.columns([.65,4.4,1.2,.9],vertical_alignment='center')
            a.write('📄')
            b.write(f"**{doc['name']}**")
            c.caption(f"{doc['chunks']} chunks")
            if d.button('Delete',key='delete_'+doc['file_hash'],help='Delete this document from the index'):
                engine.delete_document(doc['file_hash'])
                st.success('Removed from index')
                st.rerun()
            st.divider()
    if st.button('Clear chat history',key='clear_chat'):
        st.session_state.messages=[]
        st.session_state.telemetry=[]
        st.rerun()
    st.markdown('#### Supported inputs')
    st.write('✅ Selectable-text PDFs · ✅ UTF-8 .txt and .md · ❌ scanned-image OCR (not yet implemented)')

with eval_tab:
    st.markdown('### Retrieval Quality Lab')
    st.caption('Upload labelled question/source cases and measure whether the correct evidence was retrieved.')
    st.markdown('**Metrics:** Hit Rate@K = fraction of queries with a relevant result in top K. '
                'MRR = average reciprocal rank of the first relevant result.')
    st.download_button('⬇ Download example benchmark CSV',
        data=(ROOT/'eval'/'benchmark_example.csv').read_bytes(),
        file_name='benchmark_example.csv',mime='text/csv')
    benchmark=st.file_uploader('Benchmark CSV: question,expected_source,expected_page',
                              type=['csv'],key='benchmark')
    if st.button('▶ Run retrieval benchmark',type='primary',disabled=not benchmark or not stats['documents']):
        try:
            cases=list(csv.DictReader(io.StringIO(benchmark.getvalue().decode('utf-8-sig'))))
            case_limit = 25 if PUBLIC_DEMO else 100
            if len(cases) > case_limit:
                raise ValueError(f'Run at most {case_limit} cases per evaluation.')
            with st.spinner('Evaluating semantic retrieval…'):
                t0=time.perf_counter()
                metrics=evaluate_retrieval(cases,engine.search,top_k=top_k)
                elapsed=time.perf_counter()-t0
            st.session_state.last_eval=metrics
            st.session_state.last_eval_elapsed=elapsed
        except Exception as exc:
            st.error(f'Evaluation failed: {exc}')
    if 'last_eval' in st.session_state:
        m=st.session_state.last_eval
        a,b,c=st.columns(3)
        a.metric('Hit Rate@K',f'{m.hit_rate*100:.1f}%')
        b.metric('MRR',f'{m.mrr:.3f}')
        c.metric('Evaluated questions',m.count)
        st.dataframe(m.results,hide_index=True,use_container_width=True)
        st.caption('This evaluates retrieval, not whether generated answers are accurate or faithful. '
                   'Labels must be independently checked. Timing depends on hardware and dataset.')
    st.markdown('#### Query telemetry (current session)')
    if st.session_state.telemetry:
        st.dataframe(st.session_state.telemetry,hide_index=True,use_container_width=True)
        st.line_chart([round(row['retrieval_ms']) for row in st.session_state.telemetry],
                      x_label='Question number',y_label='Retrieval latency (ms)')
    else:
        st.info('Ask questions in the Chat tab to capture timing and token estimates.')
