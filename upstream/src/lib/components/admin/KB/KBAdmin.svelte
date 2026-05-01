<script>
	import { onDestroy, onMount } from 'svelte';

	let kbs = [];
	let selectedKB = null;
	let subtags = [];
	let documents = [];
	let grants = [];
	let users = [];
	let groups = [];
	let loading = true;

	// Form state
	let newKBName = '';
	let newKBDesc = '';
	let newSubtagName = '';
	let grantType = 'group';
	let grantId = '';
	let uploadSubtagId = '';
	let uploadFiles = [];
	let uploading = false;
	let uploadProgress = '';

	// UI state
	let showUsersRef = false;
	let createError = '';
	let detailError = '';

	// Per-KB rag_config (Advanced Settings) — mirrors the M5 panel from
	// ext/static/kb-admin.html. Schema must stay in sync with VALID_KEYS
	// in ext/services/kb_config.py.
	let advConfig = {};       // current values (key → coerced value or '')
	let advOpen = false;
	let advLoading = false;
	let advStatus = '';
	let advError = '';
	const ADV_SCHEMA = [
		{ key: 'rerank',                type: 'bool', label: 'Cross-encoder rerank', help: 'Default off; production uses bge-reranker-v2-m3.' },
		{ key: 'mmr',                   type: 'bool', label: 'MMR diversification' },
		{ key: 'context_expand',        type: 'bool', label: 'Context expand (siblings)' },
		{ key: 'spotlight',             type: 'bool', label: 'Spotlight wrap' },
		{ key: 'semcache',              type: 'bool', label: 'Semantic cache' },
		{ key: 'hyde',                  type: 'bool', label: 'HyDE expansion' },
		{ key: 'doc_summaries',         type: 'bool', label: 'Per-doc summaries (ingest-only)' },
		{ key: 'intent_routing',        type: 'bool', label: 'Intent routing' },
		{ key: 'top_k',                 type: 'int',  label: 'Top-K (pre-rerank pull)', min: 1, max: 200, help: 'Caps per-KB candidates pulled before rerank. Leave blank to inherit intent default. Bump for KBs whose queries enumerate many entities.' },
		{ key: 'rerank_top_k',          type: 'int',  label: 'Rerank top-K (final)', min: 1, max: 1000, help: 'Final candidate count after rerank. Default 12 from RAG_RERANK_TOP_K.' },
		{ key: 'context_expand_window', type: 'int',  label: 'Expand window', min: 0, max: 100 },
		{ key: 'hyde_n',                type: 'int',  label: 'HyDE expansions', min: 1, max: 10 },
		{ key: 'chunk_tokens',          type: 'int',  label: 'Chunk tokens (ingest)', min: 100, max: 2000 },
		{ key: 'overlap_tokens',        type: 'int',  label: 'Overlap tokens (ingest)', min: 0, max: 1000 },
		{ key: 'mmr_lambda',            type: 'float', label: 'MMR λ', min: 0, max: 1, step: 0.05, help: 'Lower = more diversity. Default 0.7 favours relevance.' }
	];

	async function loadAdvancedConfig() {
		if (!selectedKB) return;
		advLoading = true;
		advStatus = '';
		advError = '';
		try {
			const out = await api(`/api/kb/${selectedKB.id}/config`);
			const cfg = (out && out.rag_config) || {};
			advConfig = {};
			for (const f of ADV_SCHEMA) {
				if (cfg[f.key] != null) advConfig[f.key] = cfg[f.key];
				else advConfig[f.key] = (f.type === 'bool') ? false : '';
			}
		} catch (e) {
			advError = 'Could not load: ' + e.message;
		}
		advLoading = false;
	}

	async function saveAdvancedConfig() {
		if (!selectedKB) return;
		advError = '';
		advStatus = 'Saving…';
		const payload = {};
		for (const f of ADV_SCHEMA) {
			const v = advConfig[f.key];
			if (f.type === 'bool') {
				// Always send boolean state so toggling off persists.
				payload[f.key] = !!v;
			} else if (v === '' || v == null || (typeof v === 'string' && v.trim() === '')) {
				// Empty numeric → omit (inherit process default).
				continue;
			} else {
				payload[f.key] = (f.type === 'float') ? parseFloat(v) : parseInt(v, 10);
			}
		}
		try {
			// PATCH body is the rag_config keys directly (mirrors
			// ext/static/kb-admin.html and ext/routers/kb_admin.py
			// patch_rag_config — partial merge, unknown keys → 400).
			const out = await api(`/api/kb/${selectedKB.id}/config`, {
				method: 'PATCH',
				body: JSON.stringify(payload)
			});
			const cfg = (out && out.rag_config) || {};
			advConfig = {};
			for (const f of ADV_SCHEMA) {
				if (cfg[f.key] != null) advConfig[f.key] = cfg[f.key];
				else advConfig[f.key] = (f.type === 'bool') ? false : '';
			}
			advStatus = 'Saved.';
			setTimeout(() => { advStatus = ''; }, 2500);
		} catch (e) {
			advError = 'Save failed: ' + e.message;
			advStatus = '';
		}
	}

	function resetAdvancedField(key) {
		const f = ADV_SCHEMA.find((x) => x.key === key);
		if (!f) return;
		advConfig[key] = (f.type === 'bool') ? false : '';
		advConfig = advConfig;
	}

	const TOKEN = typeof localStorage !== 'undefined' ? (localStorage.token || '') : '';

	async function api(path, opts = {}) {
		const headers = {
			'Authorization': `Bearer ${TOKEN}`,
			'Content-Type': 'application/json',
			...(opts.headers || {})
		};
		if (opts.body instanceof FormData) delete headers['Content-Type'];
		const r = await fetch(path, { ...opts, headers });
		if (!r.ok) {
			const e = await r.text();
			throw new Error(e);
		}
		if (r.status === 204) return null;
		return r.json();
	}

	// KB CRUD
	async function loadKBs() {
		try {
			const resp = await api('/api/kb');
			// H2: list_kbs now returns {items, total_count}; tolerate legacy list.
			kbs = Array.isArray(resp) ? resp : (resp && resp.items) || [];
		} catch (e) {
			kbs = [];
		}
		loading = false;
	}

	async function createKB() {
		if (!newKBName.trim()) return;
		createError = '';
		try {
			await api('/api/kb', {
				method: 'POST',
				body: JSON.stringify({
					name: newKBName.trim(),
					description: newKBDesc.trim() || null
				})
			});
			newKBName = '';
			newKBDesc = '';
			await loadKBs();
		} catch (e) {
			createError = e.message;
		}
	}

	async function deleteKB(id) {
		if (!confirm('Delete this Knowledge Base and all its documents?')) return;
		try {
			await api(`/api/kb/${id}`, { method: 'DELETE' });
			if (selectedKB?.id === id) selectedKB = null;
			await loadKBs();
		} catch (e) {
			alert('Failed to delete KB: ' + e.message);
		}
	}

	// Live ingest progress — keyed by doc_id, overlaid on the table so the
	// status column reflects what the worker is doing right now (queued →
	// processing → done|failed) instead of the snapshot fetched at page
	// load. Refresh loadDocuments() on terminal events so chunk_count
	// catches up.
	let ingestProgress = {};
	let ingestSse = null;

	function closeIngestStream() {
		if (ingestSse) {
			try { ingestSse.close(); } catch (_e) { /* noop */ }
			ingestSse = null;
		}
	}

	function openIngestStream(kbId) {
		closeIngestStream();
		ingestProgress = {};
		try {
			const tok = (typeof localStorage !== 'undefined' ? localStorage.token : '') || '';
			// EventSource can't carry custom headers; the SSE endpoint
			// accepts a ?token= fallback for browser subscribers.
			const url = `/api/kb/${kbId}/ingest-stream?token=${encodeURIComponent(tok)}`;
			ingestSse = new EventSource(url);
			ingestSse.addEventListener('ingest', (ev) => {
				try {
					const payload = JSON.parse(ev.data);
					const id = Number(payload.doc_id);
					if (!Number.isFinite(id)) return;
					ingestProgress[id] = {
						stage: payload.stage || 'unknown',
						chunks: payload.chunks,
						error: payload.error
					};
					ingestProgress = ingestProgress; // svelte reactivity
					if (payload.stage === 'done' || payload.stage === 'failed') {
						loadDocuments();
					}
				} catch (_e) { /* malformed event, ignore */ }
			});
			ingestSse.addEventListener('error', () => {
				// Browser auto-reconnects on transient errors; nothing to do.
			});
		} catch (_e) {
			ingestSse = null;
		}
	}

	function liveStatus(doc) {
		const live = ingestProgress[doc.id];
		if (live && live.stage) return live.stage;
		return doc.ingest_status ?? 'unknown';
	}

	async function selectKB(kb) {
		selectedKB = kb;
		detailError = '';
		subtags = [];
		documents = [];
		grants = [];
		advConfig = {};
		advStatus = '';
		advError = '';
		openIngestStream(kb.id);
		await Promise.all([loadSubtags(), loadDocuments(), loadGrants(), loadAdvancedConfig()]);
	}

	onDestroy(closeIngestStream);

	// Subtags
	async function loadSubtags() {
		try {
			subtags = await api(`/api/kb/${selectedKB.id}/subtags`);
		} catch {
			subtags = [];
		}
	}

	async function createSubtag() {
		if (!newSubtagName.trim()) return;
		try {
			await api(`/api/kb/${selectedKB.id}/subtags`, {
				method: 'POST',
				body: JSON.stringify({ name: newSubtagName.trim() })
			});
			newSubtagName = '';
			await loadSubtags();
		} catch (e) {
			detailError = 'Subtag error: ' + e.message;
		}
	}

	async function deleteSubtag(id) {
		if (!confirm('Delete this subtag? Documents will be untagged.')) return;
		try {
			await api(`/api/kb/${selectedKB.id}/subtags/${id}`, { method: 'DELETE' });
			await loadSubtags();
		} catch (e) {
			detailError = 'Delete subtag error: ' + e.message;
		}
	}

	// Documents
	async function loadDocuments() {
		try {
			const resp = await api(`/api/kb/${selectedKB.id}/documents`);
			// H2: list_documents now returns {items, total_count}; tolerate legacy list.
			documents = Array.isArray(resp) ? resp : (resp && resp.items) || [];
			detailError = '';
		} catch (e) {
			documents = [];
			detailError = 'Load documents error: ' + e.message;
		}
	}

	function subtagName(id) {
		return subtags.find((s) => s.id === id)?.name ?? '—';
	}

	function fileTypeLabel(mime, filename) {
		if (mime) {
			if (mime.includes('wordprocessingml')) return 'docx';
			if (mime.includes('spreadsheetml')) return 'xlsx';
			if (mime === 'text/plain') return 'txt';
			if (mime === 'text/markdown') return 'md';
			if (mime === 'text/csv') return 'csv';
			if (mime === 'application/pdf') return 'pdf';
		}
		const dot = filename?.lastIndexOf('.') ?? -1;
		return dot >= 0 ? filename.slice(dot + 1) : '—';
	}

	async function deleteDocument(docId) {
		if (!confirm('Delete this document and its embeddings?')) return;
		try {
			await api(`/api/kb/${selectedKB.id}/documents/${docId}`, { method: 'DELETE' });
			await loadDocuments();
		} catch (e) {
			detailError = 'Delete document error: ' + e.message;
		}
	}

	// Upload
	function handleFileSelect(e) {
		uploadFiles = [...e.target.files];
	}

	function handleDrop(e) {
		e.preventDefault();
		uploadFiles = [...e.dataTransfer.files];
	}

	function handleDragOver(e) {
		e.preventDefault();
	}

	async function uploadAll() {
		if (!uploadSubtagId || !uploadFiles.length) return;
		uploading = true;
		uploadProgress = '';
		let ok = 0;
		let fail = 0;
		for (const file of uploadFiles) {
			uploadProgress = `Uploading ${file.name}…`;
			const fd = new FormData();
			fd.append('file', file);
			try {
				await api(`/api/kb/${selectedKB.id}/subtag/${uploadSubtagId}/upload`, {
					method: 'POST',
					body: fd
				});
				ok++;
			} catch {
				fail++;
			}
		}
		uploadProgress = `Done: ${ok} uploaded${fail ? `, ${fail} failed` : ''}`;
		uploadFiles = [];
		uploading = false;
		await loadDocuments();
	}

	// Access
	async function loadGrants() {
		try {
			grants = await api(`/api/kb/${selectedKB.id}/access`);
		} catch {
			grants = [];
		}
	}

	async function grantAccess() {
		if (!grantId.trim()) return;
		const body = grantType === 'user' ? { user_id: grantId.trim() } : { group_id: grantId.trim() };
		try {
			await api(`/api/kb/${selectedKB.id}/access`, {
				method: 'POST',
				body: JSON.stringify(body)
			});
			grantId = '';
			await loadGrants();
		} catch (e) {
			detailError = 'Grant access error: ' + e.message;
		}
	}

	async function revokeAccess(id) {
		try {
			await api(`/api/kb/${selectedKB.id}/access/${id}`, { method: 'DELETE' });
			await loadGrants();
		} catch (e) {
			detailError = 'Revoke error: ' + e.message;
		}
	}

	// Users & Groups (for reference)
	async function loadUsersAndGroups() {
		try {
			const r = await api('/api/v1/users/all');
			users = Array.isArray(r) ? r : (r?.users ?? []);
		} catch {
			users = [];
		}
		try {
			groups = await api('/api/v1/groups/');
		} catch {
			groups = [];
		}
	}

	function userLabel(id) {
		const u = users.find((x) => x.id === id);
		return u ? `${u.email} (${u.name})` : id;
	}

	function groupLabel(id) {
		const g = groups.find((x) => x.id === id);
		return g ? g.name : id;
	}

	function statusColor(status) {
		if (status === 'done' || status === 'ready') return 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300';
		if (status === 'pending' || status === 'processing') return 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-300';
		if (status === 'failed' || status === 'error') return 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300';
		return 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400';
	}

	onMount(() => {
		loadKBs();
		loadUsersAndGroups();
	});
</script>

<div class="min-h-screen bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100">
	<!-- Header -->
	<div class="px-6 py-5 border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-850">
		<div class="flex flex-col gap-4 max-w-7xl mx-auto">
			<div class="flex items-center justify-between">
				<h1 class="text-xl font-semibold">Knowledge Base Management</h1>
				<span class="text-sm text-gray-500 dark:text-gray-400">{kbs.length} Knowledge Base{kbs.length !== 1 ? 's' : ''}</span>
			</div>

			<!-- Create KB form -->
			<div class="flex flex-wrap items-end gap-2">
				<div class="flex flex-col gap-1">
					<label class="text-xs font-medium text-gray-600 dark:text-gray-400">Name</label>
					<input
						class="bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg px-3 py-2 text-sm w-48 focus:outline-none focus:ring-2 focus:ring-blue-500"
						type="text"
						placeholder="KB name…"
						bind:value={newKBName}
						on:keydown={(e) => e.key === 'Enter' && createKB()}
					/>
				</div>
				<div class="flex flex-col gap-1">
					<label class="text-xs font-medium text-gray-600 dark:text-gray-400">Description (optional)</label>
					<input
						class="bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg px-3 py-2 text-sm w-64 focus:outline-none focus:ring-2 focus:ring-blue-500"
						type="text"
						placeholder="Brief description…"
						bind:value={newKBDesc}
						on:keydown={(e) => e.key === 'Enter' && createKB()}
					/>
				</div>
				<button
					class="px-3 py-2 text-sm rounded-lg font-medium bg-blue-600 text-white hover:bg-blue-700 transition disabled:opacity-50"
					on:click={createKB}
					disabled={!newKBName.trim()}
				>
					Create KB
				</button>
				{#if createError}
					<p class="text-xs text-red-500">{createError}</p>
				{/if}
			</div>
		</div>
	</div>

	<!-- Main content -->
	<div class="max-w-7xl mx-auto px-6 py-6">
		{#if loading}
			<div class="flex items-center justify-center py-16">
				<div class="flex flex-col items-center gap-3 text-gray-500 dark:text-gray-400">
					<svg class="animate-spin size-8" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
						<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
						<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
					</svg>
					<span class="text-sm">Loading knowledge bases…</span>
				</div>
			</div>
		{:else}
			<div class="flex gap-6">
				<!-- Left sidebar: KB list -->
				<div class="w-64 flex-none flex flex-col gap-2">
					<h2 class="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400 px-1">Knowledge Bases</h2>

					{#if kbs.length === 0}
						<div class="bg-white dark:bg-gray-850 rounded-xl border border-gray-200 dark:border-gray-800 p-4 text-sm text-gray-500 dark:text-gray-400 text-center">
							No KBs yet. Create one above.
						</div>
					{:else}
						{#each kbs as kb (kb.id)}
							<div
								class="bg-white dark:bg-gray-850 rounded-xl border transition cursor-pointer {selectedKB?.id === kb.id
									? 'border-blue-500 dark:border-blue-500 ring-1 ring-blue-500'
									: 'border-gray-200 dark:border-gray-800 hover:border-gray-300 dark:hover:border-gray-700'}"
							>
								<button
									class="w-full text-left px-3 py-3 focus:outline-none"
									on:click={() => selectKB(kb)}
								>
									<div class="font-medium text-sm truncate">{kb.name}</div>
									{#if kb.description}
										<div class="text-xs text-gray-500 dark:text-gray-400 mt-0.5 truncate">{kb.description}</div>
									{/if}
									<div class="text-xs text-gray-400 dark:text-gray-500 mt-1">
										{kb.document_count ?? 0} doc{(kb.document_count ?? 0) !== 1 ? 's' : ''}
									</div>
								</button>
								<div class="px-3 pb-2 flex justify-end">
									<button
										class="px-2 py-1 text-xs rounded-md font-medium bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 hover:bg-red-200 dark:hover:bg-red-900/50 transition"
										on:click|stopPropagation={() => deleteKB(kb.id)}
									>
										Delete
									</button>
								</div>
							</div>
						{/each}
					{/if}
				</div>

				<!-- Right panel: KB detail -->
				<div class="flex-1 min-w-0">
					{#if !selectedKB}
						<div class="bg-white dark:bg-gray-850 rounded-xl border border-gray-200 dark:border-gray-800 p-12 flex flex-col items-center justify-center gap-3 text-gray-400 dark:text-gray-500">
							<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="size-12 opacity-40">
								<path d="M11.25 4.533A9.707 9.707 0 006 3a9.735 9.735 0 00-3.25.555.75.75 0 00-.5.707v14.25a.75.75 0 001 .707A8.237 8.237 0 016 18.75c1.995 0 3.823.707 5.25 1.886V4.533zM12.75 20.636A8.214 8.214 0 0118 18.75c.966 0 1.89.166 2.75.47a.75.75 0 001-.708V4.262a.75.75 0 00-.5-.707A9.735 9.735 0 0018 3a9.707 9.707 0 00-5.25 1.533v16.103z" />
							</svg>
							<p class="text-sm font-medium">Select a Knowledge Base to manage it</p>
						</div>
					{:else}
						<div class="flex flex-col gap-5">
							<!-- KB title bar -->
							<div class="bg-white dark:bg-gray-850 rounded-xl border border-gray-200 dark:border-gray-800 px-5 py-4">
								<div class="flex items-center justify-between">
									<div>
										<h2 class="text-lg font-semibold">{selectedKB.name}</h2>
										{#if selectedKB.description}
											<p class="text-sm text-gray-500 dark:text-gray-400 mt-0.5">{selectedKB.description}</p>
										{/if}
									</div>
									<span class="text-xs text-gray-400 dark:text-gray-500">ID: {selectedKB.id}</span>
								</div>
								{#if detailError}
									<p class="mt-2 text-xs text-red-500">{detailError}</p>
								{/if}
							</div>

							<!-- Advanced Settings (per-KB rag_config). Collapsed by default. -->
							<div class="bg-white dark:bg-gray-850 rounded-xl border border-gray-200 dark:border-gray-800">
								<button
									class="w-full flex items-center justify-between px-5 py-3 text-left focus:outline-none"
									on:click={() => { advOpen = !advOpen; }}
									aria-expanded={advOpen}
									aria-controls="adv-section-{selectedKB.id}"
								>
									<span class="text-sm font-semibold flex items-center gap-2">
										<span class="text-blue-500">{advOpen ? '▼' : '▶'}</span>
										Advanced Settings (per-KB retrieval overrides)
									</span>
									<span class="text-xs text-gray-400 dark:text-gray-500">
										{Object.values(advConfig).filter((v) => v !== '' && v !== false && v != null).length} active
									</span>
								</button>
								{#if advOpen}
									<div id="adv-section-{selectedKB.id}" class="px-5 pb-5 border-t border-gray-200 dark:border-gray-800">
										<p class="text-xs text-gray-500 dark:text-gray-400 mt-3 mb-4">
											Per-KB overrides for retrieval flags. Empty/unchecked = inherit process default. Saved values override env vars at request time. Schema mirrored from <code>ext/services/kb_config.py</code>.
										</p>

										{#if advLoading}
											<p class="text-xs text-gray-400">Loading…</p>
										{:else}
											<div class="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-4">
												{#each ADV_SCHEMA as f (f.key)}
													<div class="flex flex-col gap-1">
														{#if f.type === 'bool'}
															<label class="inline-flex items-center gap-2 text-sm font-medium text-gray-700 dark:text-gray-300">
																<input
																	type="checkbox"
																	bind:checked={advConfig[f.key]}
																	class="rounded border-gray-300 dark:border-gray-700"
																/>
																{f.label}
															</label>
														{:else if f.type === 'int'}
															<label class="text-xs font-medium text-gray-600 dark:text-gray-400" for="adv-{f.key}-{selectedKB.id}">
																{f.label}
																<span class="text-gray-400 dark:text-gray-500 font-normal">({f.min}–{f.max})</span>
															</label>
															<input
																id="adv-{f.key}-{selectedKB.id}"
																type="number"
																min={f.min}
																max={f.max}
																step="1"
																placeholder="inherit"
																bind:value={advConfig[f.key]}
																class="bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
															/>
														{:else if f.type === 'float'}
															<label class="text-xs font-medium text-gray-600 dark:text-gray-400" for="adv-{f.key}-{selectedKB.id}">
																{f.label}
																<span class="text-gray-400 dark:text-gray-500 font-normal">({advConfig[f.key] !== '' && advConfig[f.key] != null ? advConfig[f.key] : '—'})</span>
															</label>
															<input
																id="adv-{f.key}-{selectedKB.id}"
																type="range"
																min={f.min}
																max={f.max}
																step={f.step || 0.05}
																bind:value={advConfig[f.key]}
																class="w-full"
															/>
														{/if}
														{#if f.help}
															<span class="text-xs text-gray-400 dark:text-gray-500">{f.help}</span>
														{/if}
													</div>
												{/each}
											</div>

											<div class="flex items-center gap-3 mt-5">
												<button
													class="px-4 py-1.5 text-sm rounded-lg font-medium bg-blue-600 text-white hover:bg-blue-700 transition"
													on:click={saveAdvancedConfig}
												>
													Save
												</button>
												<button
													class="px-4 py-1.5 text-sm rounded-lg font-medium bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700 transition"
													on:click={loadAdvancedConfig}
												>
													Reload
												</button>
												{#if advStatus}
													<span class="text-xs text-green-600 dark:text-green-400">{advStatus}</span>
												{/if}
												{#if advError}
													<span class="text-xs text-red-500">{advError}</span>
												{/if}
											</div>
										{/if}
									</div>
								{/if}
							</div>

							<!-- Two columns: subtags + upload -->
							<div class="grid grid-cols-1 lg:grid-cols-2 gap-5">
								<!-- Subtags -->
								<div class="bg-white dark:bg-gray-850 rounded-xl border border-gray-200 dark:border-gray-800 p-4">
									<h3 class="text-sm font-semibold mb-3">Subtags</h3>

									<!-- Add subtag -->
									<div class="flex gap-2 mb-3">
										<input
											class="bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg px-3 py-1.5 text-sm flex-1 focus:outline-none focus:ring-2 focus:ring-blue-500"
											type="text"
											placeholder="New subtag name…"
											bind:value={newSubtagName}
											on:keydown={(e) => e.key === 'Enter' && createSubtag()}
										/>
										<button
											class="px-3 py-1.5 text-sm rounded-lg font-medium bg-blue-600 text-white hover:bg-blue-700 transition disabled:opacity-50"
											on:click={createSubtag}
											disabled={!newSubtagName.trim()}
										>
											Add
										</button>
									</div>

									{#if subtags.length === 0}
										<p class="text-xs text-gray-400 dark:text-gray-500 text-center py-4">No subtags yet.</p>
									{:else}
										<div class="flex flex-wrap gap-2">
											{#each subtags as subtag (subtag.id)}
												<span class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300">
													{subtag.name}
													<button
														class="hover:text-red-500 dark:hover:text-red-400 transition ml-0.5"
														on:click={() => deleteSubtag(subtag.id)}
														title="Delete subtag"
													>
														<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" class="size-3">
															<path d="M5.28 4.22a.75.75 0 00-1.06 1.06L6.94 8l-2.72 2.72a.75.75 0 101.06 1.06L8 9.06l2.72 2.72a.75.75 0 101.06-1.06L9.06 8l2.72-2.72a.75.75 0 00-1.06-1.06L8 6.94 5.28 4.22z" />
														</svg>
													</button>
												</span>
											{/each}
										</div>
									{/if}
								</div>

								<!-- Upload -->
								<div class="bg-white dark:bg-gray-850 rounded-xl border border-gray-200 dark:border-gray-800 p-4">
									<h3 class="text-sm font-semibold mb-3">Upload Documents</h3>

									<!-- Subtag selector -->
									<div class="mb-3">
										<label class="text-xs font-medium text-gray-600 dark:text-gray-400 block mb-1">Target Subtag</label>
										<select
											class="bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg px-3 py-1.5 text-sm w-full focus:outline-none focus:ring-2 focus:ring-blue-500"
											bind:value={uploadSubtagId}
										>
											<option value="">— select subtag —</option>
											{#each subtags as subtag (subtag.id)}
												<option value={subtag.id}>{subtag.name}</option>
											{/each}
										</select>
										{#if subtags.length === 0}
											<p class="text-xs text-yellow-600 dark:text-yellow-400 mt-1">Create a subtag first.</p>
										{/if}
									</div>

									<!-- Drop zone -->
									<div
										class="border-2 border-dashed border-gray-300 dark:border-gray-700 rounded-xl p-6 text-center transition hover:border-blue-400 dark:hover:border-blue-600 cursor-pointer"
										on:drop={handleDrop}
										on:dragover={handleDragOver}
									>
										<input
											id="file-upload-{selectedKB.id}"
											type="file"
											multiple
											accept=".txt,.md,.pdf,.docx,.xlsx,.csv"
											class="hidden"
											on:change={handleFileSelect}
										/>
										<label for="file-upload-{selectedKB.id}" class="cursor-pointer">
											<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="size-8 mx-auto mb-2 text-gray-400 dark:text-gray-500">
												<path fill-rule="evenodd" d="M11.47 2.47a.75.75 0 011.06 0l4.5 4.5a.75.75 0 01-1.06 1.06l-3.22-3.22V16.5a.75.75 0 01-1.5 0V4.81L8.03 8.03a.75.75 0 01-1.06-1.06l4.5-4.5zM3 15.75a.75.75 0 01.75.75v2.25a1.5 1.5 0 001.5 1.5h13.5a1.5 1.5 0 001.5-1.5V16.5a.75.75 0 011.5 0v2.25a3 3 0 01-3 3H5.25a3 3 0 01-3-3V16.5a.75.75 0 01.75-.75z" clip-rule="evenodd" />
											</svg>
											<p class="text-sm text-gray-500 dark:text-gray-400">
												Drop files here or <span class="text-blue-600 dark:text-blue-400 font-medium">click to browse</span>
											</p>
											<p class="text-xs text-gray-400 dark:text-gray-500 mt-1">TXT, MD, PDF, DOCX, XLSX, CSV</p>
										</label>
									</div>

									<!-- Selected files -->
									{#if uploadFiles.length > 0}
										<div class="mt-3 flex flex-wrap gap-1.5">
											{#each uploadFiles as file (file.name)}
												<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-medium bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300">
													{file.name}
												</span>
											{/each}
										</div>
									{/if}

									<!-- Upload button + progress -->
									<div class="mt-3 flex items-center gap-3">
										<button
											class="px-3 py-1.5 text-sm rounded-lg font-medium bg-blue-600 text-white hover:bg-blue-700 transition disabled:opacity-50"
											on:click={uploadAll}
											disabled={uploading || !uploadSubtagId || uploadFiles.length === 0}
										>
											{uploading ? 'Uploading…' : `Upload ${uploadFiles.length} file${uploadFiles.length !== 1 ? 's' : ''}`}
										</button>
										{#if uploadProgress}
											<span class="text-xs text-gray-500 dark:text-gray-400">{uploadProgress}</span>
										{/if}
									</div>
								</div>
							</div>

							<!-- Documents table -->
							<div class="bg-white dark:bg-gray-850 rounded-xl border border-gray-200 dark:border-gray-800 p-4">
								<div class="flex items-center justify-between mb-3">
									<h3 class="text-sm font-semibold">Documents</h3>
									<span class="text-xs text-gray-400 dark:text-gray-500">{documents.length} total</span>
								</div>

								{#if documents.length === 0}
									<p class="text-xs text-gray-400 dark:text-gray-500 text-center py-6">No documents uploaded yet.</p>
								{:else}
									<div class="overflow-x-auto">
										<table class="w-full text-sm">
											<thead>
												<tr class="border-b border-gray-200 dark:border-gray-700">
													<th class="text-left py-2 pr-4 text-xs font-semibold text-gray-500 dark:text-gray-400">Filename</th>
													<th class="text-left py-2 pr-4 text-xs font-semibold text-gray-500 dark:text-gray-400">Type</th>
													<th class="text-left py-2 pr-4 text-xs font-semibold text-gray-500 dark:text-gray-400">Subtag</th>
													<th class="text-right py-2 pr-4 text-xs font-semibold text-gray-500 dark:text-gray-400">Chunks</th>
													<th class="text-left py-2 pr-4 text-xs font-semibold text-gray-500 dark:text-gray-400">Status</th>
													<th class="text-right py-2 text-xs font-semibold text-gray-500 dark:text-gray-400">Actions</th>
												</tr>
											</thead>
											<tbody>
												{#each documents as doc (doc.id)}
													<tr class="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-800/40 transition">
														<td class="py-2 pr-4 font-medium truncate max-w-[180px]" title={doc.filename ?? doc.name}>
															{doc.filename ?? doc.name ?? '—'}
														</td>
														<td class="py-2 pr-4 text-gray-500 dark:text-gray-400 uppercase text-xs">
															{fileTypeLabel(doc.mime_type, doc.filename)}
														</td>
														<td class="py-2 pr-4">
															<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-medium bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300">
																{subtagName(doc.subtag_id)}
															</span>
														</td>
														<td class="py-2 pr-4 text-right text-gray-600 dark:text-gray-300">
															{doc.chunk_count ?? 0}
														</td>
														<td class="py-2 pr-4">
															<span class="inline-flex items-center px-2 py-0.5 rounded-md text-xs font-medium {statusColor(liveStatus(doc))}">
																{liveStatus(doc)}
															</span>
														</td>
														<td class="py-2 text-right">
															<button
																class="px-2 py-1 text-xs rounded-md font-medium bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 hover:bg-red-200 dark:hover:bg-red-900/50 transition"
																on:click={() => deleteDocument(doc.id)}
															>
																Delete
															</button>
														</td>
													</tr>
												{/each}
											</tbody>
										</table>
									</div>
								{/if}
							</div>

							<!-- Access control -->
							<div class="bg-white dark:bg-gray-850 rounded-xl border border-gray-200 dark:border-gray-800 p-4">
								<h3 class="text-sm font-semibold mb-3">Access Control (RBAC)</h3>

								<!-- Grant form -->
								<div class="flex flex-wrap items-end gap-2 mb-4 pb-4 border-b border-gray-100 dark:border-gray-800">
									<div class="flex flex-col gap-1">
										<label class="text-xs font-medium text-gray-600 dark:text-gray-400">Grant type</label>
										<select
											class="bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
											bind:value={grantType}
										>
											<option value="group">Group</option>
											<option value="user">User</option>
										</select>
									</div>
									<div class="flex flex-col gap-1">
										<label class="text-xs font-medium text-gray-600 dark:text-gray-400">
											{grantType === 'user' ? 'User' : 'Group'}
										</label>
										<select
											class="bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg px-3 py-1.5 text-sm w-72 focus:outline-none focus:ring-2 focus:ring-blue-500"
											bind:value={grantId}
										>
											<option value="">— select {grantType} —</option>
											{#if grantType === 'user'}
												{#each users as u (u.id)}
													<option value={u.id}>{u.email} ({u.name})</option>
												{/each}
											{:else}
												{#each groups as g (g.id)}
													<option value={g.id}>{g.name}{g.member_count !== undefined ? ` · ${g.member_count} member${g.member_count === 1 ? '' : 's'}` : ''}</option>
												{/each}
											{/if}
										</select>
									</div>
									<button
										class="px-3 py-1.5 text-sm rounded-lg font-medium bg-blue-600 text-white hover:bg-blue-700 transition disabled:opacity-50"
										on:click={grantAccess}
										disabled={!grantId.trim()}
									>
										Grant Access
									</button>
								</div>

								<!-- Current grants -->
								{#if grants.length === 0}
									<p class="text-xs text-gray-400 dark:text-gray-500 text-center py-4">No grants yet. This KB is private to admins.</p>
								{:else}
									<div class="flex flex-col gap-2">
										{#each grants as grant (grant.id)}
											<div class="flex items-center justify-between py-2 px-3 rounded-lg bg-gray-50 dark:bg-gray-900/50">
												<div class="flex items-center gap-2">
													{#if grant.group_id}
														<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-medium bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300">
															Group
														</span>
														<span class="text-sm text-gray-700 dark:text-gray-300">{groupLabel(grant.group_id)}</span>
													{:else}
														<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-medium bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300">
															User
														</span>
														<span class="text-sm text-gray-700 dark:text-gray-300">{userLabel(grant.user_id)}</span>
													{/if}
												</div>
												<button
													class="px-2 py-1 text-xs rounded-md font-medium bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 hover:bg-red-200 dark:hover:bg-red-900/50 transition"
													on:click={() => revokeAccess(grant.id)}
												>
													Revoke
												</button>
											</div>
										{/each}
									</div>
								{/if}
							</div>
						</div>
					{/if}
				</div>
			</div>

			<!-- Users & Groups reference (collapsible) -->
			<div class="mt-6 bg-white dark:bg-gray-850 rounded-xl border border-gray-200 dark:border-gray-800">
				<button
					class="w-full flex items-center justify-between px-5 py-4 text-left"
					on:click={() => (showUsersRef = !showUsersRef)}
				>
					<span class="text-sm font-semibold">Users &amp; Groups Reference</span>
					<div class="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
						<span>{users.length} user{users.length !== 1 ? 's' : ''}, {groups.length} group{groups.length !== 1 ? 's' : ''}</span>
						<svg
							xmlns="http://www.w3.org/2000/svg"
							viewBox="0 0 16 16"
							fill="currentColor"
							class="size-4 transition-transform {showUsersRef ? 'rotate-180' : ''}"
						>
							<path fill-rule="evenodd" d="M4.22 6.22a.75.75 0 011.06 0L8 8.94l2.72-2.72a.75.75 0 111.06 1.06l-3.25 3.25a.75.75 0 01-1.06 0L4.22 7.28a.75.75 0 010-1.06z" clip-rule="evenodd" />
						</svg>
					</div>
				</button>

				{#if showUsersRef}
					<div class="px-5 pb-5 grid grid-cols-1 lg:grid-cols-2 gap-5 border-t border-gray-100 dark:border-gray-800 pt-4">
						<!-- Users table -->
						<div>
							<h4 class="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-2">Users</h4>
							{#if users.length === 0}
								<p class="text-xs text-gray-400 dark:text-gray-500">No users loaded.</p>
							{:else}
								<div class="overflow-x-auto">
									<table class="w-full text-xs">
										<thead>
											<tr class="border-b border-gray-200 dark:border-gray-700">
												<th class="text-left py-1.5 pr-3 font-semibold text-gray-500 dark:text-gray-400">Email</th>
												<th class="text-left py-1.5 pr-3 font-semibold text-gray-500 dark:text-gray-400">Role</th>
												<th class="text-left py-1.5 font-semibold text-gray-500 dark:text-gray-400">ID</th>
											</tr>
										</thead>
										<tbody>
											{#each users as u (u.id)}
												<tr class="border-b border-gray-100 dark:border-gray-800">
													<td class="py-1.5 pr-3 text-gray-700 dark:text-gray-300">{u.email}</td>
													<td class="py-1.5 pr-3">
														<span class="inline-flex items-center px-1.5 py-0.5 rounded text-xs font-medium {u.role === 'admin' ? 'bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-300' : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400'}">
															{u.role}
														</span>
													</td>
													<td class="py-1.5 font-mono text-gray-400 dark:text-gray-500 text-xs truncate max-w-[120px]" title={u.id}>
														{u.id}
													</td>
												</tr>
											{/each}
										</tbody>
									</table>
								</div>
							{/if}
						</div>

						<!-- Groups table -->
						<div>
							<h4 class="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-2">Groups</h4>
							{#if groups.length === 0}
								<p class="text-xs text-gray-400 dark:text-gray-500">No groups loaded.</p>
							{:else}
								<div class="overflow-x-auto">
									<table class="w-full text-xs">
										<thead>
											<tr class="border-b border-gray-200 dark:border-gray-700">
												<th class="text-left py-1.5 pr-3 font-semibold text-gray-500 dark:text-gray-400">Name</th>
												<th class="text-left py-1.5 font-semibold text-gray-500 dark:text-gray-400">ID</th>
											</tr>
										</thead>
										<tbody>
											{#each groups as g (g.id)}
												<tr class="border-b border-gray-100 dark:border-gray-800">
													<td class="py-1.5 pr-3 text-gray-700 dark:text-gray-300">{g.name}</td>
													<td class="py-1.5 font-mono text-gray-400 dark:text-gray-500 text-xs truncate max-w-[150px]" title={g.id}>
														{g.id}
													</td>
												</tr>
											{/each}
										</tbody>
									</table>
								</div>
							{/if}
						</div>
					</div>
				{/if}
			</div>
		{/if}
	</div>
</div>
