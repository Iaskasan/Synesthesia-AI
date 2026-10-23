'use strict';
const $ = (selector) => document.querySelector(selector);
let csrf = '', currentProject = null, registering = false, previewUrl = null, poll = null, snapshot = null;
function node(tag, text, cls) { const el = document.createElement(tag); if (text !== undefined) el.textContent = text; if (cls) el.className = cls; return el; }
function notice(message, error = false) { const el = $('#notice'); el.textContent = message; el.className = error ? 'error' : ''; el.hidden = !message; }
async function api(path, options = {}) {
  const headers = { 'X-CSRF-Token': csrf, ...options.headers };
  if (options.body && !(options.body instanceof FormData)) { headers['Content-Type'] = 'application/json'; options.body = JSON.stringify(options.body); }
  const response = await fetch('/api' + path, { ...options, headers });
  if (!response.ok) {
    let message = 'Request failed. Please try again.';
    try { const data = await response.json(); message = typeof data.detail === 'string' ? data.detail : data.detail?.map(x => x.msg).join(';') || message; } catch (_) { /* Non-JSON infrastructure response. */ }
    if (response.status === 401 && csrf) { csrf = ''; currentProject = null; clearTimeout(poll); show('welcome'); $('#identity').textContent = ''; $('#logout').hidden = true; }
    throw new Error(message);
  }
  return response.status === 204 ? null : response.json();
}
async function action(button, work) { button.disabled = true; notice(''); try { await work(); } catch (error) { notice(error.message, true); } finally { button.disabled = false; } }
function show(id) { for (const section of ['welcome', 'workspace', 'studio']) $('#' + section).hidden = section !== id; }
function signedIn(data) { csrf = data.csrf; $('#identity').textContent = data.email; $('#logout').hidden = false; $('#auth-form').reset(); }
$('#auth-toggle').onclick = () => { registering = !registering; $('#auth-title').textContent = registering ? 'Make room for ideas.' : 'Welcome back.'; $('#auth-caption').textContent = registering ? 'Create your private Synesthesia account.' : 'Sign in to pick up where you left off.'; $('#auth-submit').textContent = registering ? 'Create account ↗' : 'Sign in ↗'; $('#auth-toggle').textContent = registering ? 'Already a member? Sign in' : 'New here? Create an account'; $('#auth-form [name=password]').autocomplete = registering ? 'new-password' : 'current-password'; };
$('#auth-form').onsubmit = (event) => { event.preventDefault(); action($('#auth-submit'), async () => { const data = Object.fromEntries(new FormData(event.target)); signedIn(await api(registering ? '/register' : '/login', { method: 'POST', body: data })); await dashboard(); }); };
$('#logout').onclick = () => action($('#logout'), async () => { await api('/logout', { method: 'POST' }); csrf = ''; currentProject = null; clearTimeout(poll); $('#identity').textContent = ''; $('#logout').hidden = true; show('welcome'); });
async function dashboard() {
  currentProject = null; snapshot = null; clearTimeout(poll);
  const projects = await api('/projects'); show('workspace'); $('#projects').replaceChildren(); $('#project-count').textContent = `${projects.length} project${projects.length === 1 ? '' : 's'}`;
  if (!projects.length) $('#projects').append(node('div', 'Your collection starts with a sound. Create your first project above.', 'empty'));
  for (const project of projects) {
    const card = node('button', undefined, 'project-card'); card.append(node('span', '∿', 'project-symbol'), node('h3', project.title), node('small', new Date(project.created_at * 1000).toLocaleDateString() + ' · Open studio ↗'));
    card.onclick = () => action(card, () => openProject(project.id)); $('#projects').append(card);
  }
}
$('#project-form').onsubmit = (event) => { event.preventDefault(); action(event.target.querySelector('button'), async () => { const project = await api('/projects', { method: 'POST', body: { title: $('#project-title').value } }); event.target.reset(); await openProject(project.id); }); };
$('#back').onclick = () => action($('#back'), dashboard);
$('#rename').onclick = () => { const title = window.prompt('Project name', $('#studio-title').textContent); if (title?.trim()) action($('#rename'), async () => { await api(`/projects/${currentProject}`, { method: 'PATCH', body: { title } }); await refreshProject(); }); };
$('#delete').onclick = () => { if (window.confirm('Delete this project, its analyses and generated images? This cannot be undone.')) action($('#delete'), async () => { await api(`/projects/${currentProject}`, { method: 'DELETE' }); await dashboard(); }); };
async function openProject(id) {
  clearTimeout(poll); currentProject = id; snapshot = null;
  $('#creative-form').reset(); $('#creative-fields').disabled = true; $('#analysis-select').replaceChildren(); $('#upload-form').reset(); clearAudioPreview();
  const moods = await api('/moods'); $('#mood-select').replaceChildren(...moods.map(label => { const el = node('option', label); el.value = label; return el; }));
  show('studio'); await refreshProject();
}
function clearAudioPreview() { if (previewUrl) URL.revokeObjectURL(previewUrl); previewUrl = null; $('#audio-preview').removeAttribute('src'); $('#audio-preview').hidden = true; }
$('#audio-file').onchange = () => { clearAudioPreview(); const file = $('#audio-file').files[0]; if (file) { previewUrl = URL.createObjectURL(file); $('#audio-preview').src = previewUrl; $('#audio-preview').hidden = false; } };
$('#upload-form').onsubmit = (event) => { event.preventDefault(); action(event.target.querySelector('button'), async () => { const form = new FormData(event.target); if (form.get('file').size > 25 * 1024 * 1024) throw new Error('Choose an audio file smaller than 25 MB.'); form.set('retain', event.target.elements.retain.checked ? 'true' : 'false'); await api(`/projects/${currentProject}/audio`, { method: 'POST', body: form }); event.target.reset(); clearAudioPreview(); notice('Analysis queued. The local GPU worker will process your excerpt.'); await refreshProject(); }); };
function chooseSuggestedMoods() {
  const analysis = snapshot?.analyses.find(a => a.id === $('#analysis-select').value);
  const detected = analysis?.predictions.filter(p => p.detected) || [];
  const selected = (detected.length ? detected : analysis?.predictions || []).slice(0, 3).map(p => p.label);
  for (const option of $('#mood-select').options) option.selected = selected.includes(option.value);
}
$('#analysis-select').onchange = chooseSuggestedMoods;
function creativeInput() { const form = new FormData($('#creative-form')); const moods = Array.from($('#mood-select').selectedOptions, o => o.value); if (!moods.length || moods.length > 5) throw new Error('Choose between 1 and 5 moods.'); return { moods, style: form.get('style'), subject: form.get('subject'), abstraction: Number(form.get('abstraction')), music_influence: Number(form.get('music_influence')) }; }
$('#compose').onclick = () => action($('#compose'), async () => { const result = await api(`/analyses/${$('#analysis-select').value}/prompt`, { method: 'POST', body: creativeInput() }); $('#creative-form [name=prompt]').value = result.prompt; });
$('#creative-form').onsubmit = (event) => { event.preventDefault(); action(event.target.querySelector('[type=submit]'), async () => { const [width, height] = $('#dimensions').value.split(',').map(Number); await api(`/analyses/${$('#analysis-select').value}/generations`, { method: 'POST', body: { ...creativeInput(), prompt: event.target.elements.prompt.value, seed: Number(event.target.elements.seed.value), width, height } }); notice('Artwork queued. You can keep exploring while your image is generated.'); await refreshProject(); }); };
async function refreshProject() {
  clearTimeout(poll); const id = currentProject; if (!id) return;
  const data = await api(`/projects/${id}`); if (currentProject !== id) return; snapshot = data; $('#studio-title').textContent = data.title;
  $('#analyses').replaceChildren();
  if (!data.analyses.length) $('#analyses').append(node('p', 'Your mood interpretation will appear here after analysis.', 'muted'));
  for (const analysis of data.analyses) {
    const card = node('div', undefined, 'analysis-card'); card.append(node('h3', analysis.filename), node('span', analysis.status, 'status ' + analysis.status));
    if (analysis.error) card.append(node('p', analysis.error, 'fine'));
    if (analysis.status === 'queued') card.append(node('p', 'Waiting for the local worker.', 'fine'));
    if (analysis.features) {
      const stats = node('div', undefined, 'stats');
      for (const [value, title] of [[analysis.features.tempo_bpm.toFixed(0), 'TEMPO · BPM'], [analysis.features.rms_energy.toFixed(3), 'ENERGY · RMS'], [analysis.features.spectral_centroid_hz.toFixed(0), 'BRIGHTNESS · HZ']]) { const stat = node('div', undefined, 'stat'); stat.append(node('strong', value), node('small', title)); stats.append(stat); } card.append(stats);
      for (const prediction of analysis.predictions) { const row = node('div', undefined, 'mood-row'); const bar = node('progress'); bar.max = 1; bar.value = prediction.confidence; bar.setAttribute('aria-label', prediction.label + ' confidence'); row.append(node('span', prediction.label + (prediction.detected ? ' ·' : '')), bar, node('small', Math.round(prediction.confidence * 100) + '%')); card.append(row); }
      card.append(node('p', 'Scores describe a model interpretation, not objective emotion. A dot marks a detected mood.', 'fine'));
    }
    $('#analyses').append(card);
  }
  const success = data.analyses.filter(a => a.status === 'succeeded'), previous = $('#analysis-select').value;
  $('#analysis-select').replaceChildren(...success.map(a => { const option = node('option', `${a.filename} · ${a.start_seconds}s–${(a.start_seconds + a.duration_seconds).toFixed(1)}s`); option.value = a.id; return option; }));
  if (success.some(a => a.id === previous)) $('#analysis-select').value = previous;
  else if (success.length) chooseSuggestedMoods();
  $('#creative-fields').disabled = !success.length; $('#creative-hint').hidden = !!success.length;
  $('#generations').replaceChildren();
  if (!data.generations.length) $('#generations').append(node('div', 'A blank canvas, for now. Your generated artwork will live here.', 'empty'));
  for (const generation of data.generations) {
    const card = node('article', undefined, 'generation-card');
    if (generation.image_url) { const image = node('img'); image.src = generation.image_url; image.alt = 'AI-generated artwork: ' + generation.prompt; image.loading = 'lazy'; card.append(image); }
    else card.append(node('div', generation.status === 'failed' ? 'Try a new interpretation' : 'A world in the making…', 'generation-placeholder'));
    const body = node('div', undefined, 'generation-body'); body.append(node('span', generation.status, 'status ' + generation.status), node('p', generation.error || generation.prompt));
    const links = node('div', undefined, 'button-row');
    if (generation.image_url) { const link = node('a', 'Download image ↗'); link.href = generation.image_url; link.download = 'synesthesia.png'; links.append(link); }
    const metadata = node('a', 'Export settings ↗'); metadata.href = `/api/generations/${generation.id}/metadata`; links.append(metadata); body.append(links); card.append(body); $('#generations').append(card);
  }
  if ([...data.analyses, ...data.generations].some(item => ['queued', 'running'].includes(item.status))) poll = setTimeout(() => refreshProject().catch(error => notice(error.message, true)), 4000);
}
(async () => { try { signedIn(await api('/me')); await dashboard(); } catch (error) { show('welcome'); if (error.message !== 'Please sign in.') notice(error.message, true); } })();
