<script lang="ts">
	import { onMount } from 'svelte';
	import { getAvailableKBs, getKBSubtags, setChatKBConfig, type KBSelection } from '$lib/apis/kb';

	export let token: string = '';
	export let chatId: string = '';
	export let selections: KBSelection[] = [];
	export let locked: boolean = false;

	let kbs: any[] = [];
	let open = false;
	let loading = true;

	// Per-KB expand state + cached subtags
	let expanded: Record<number, boolean> = {};
	let subtagsByKB: Record<number, any[]> = {};
	let subtagsLoading: Record<number, boolean> = {};

	onMount(async () => {
		if (!token && typeof localStorage !== 'undefined') {
			token = localStorage.token || '';
		}
		await loadKBs();
	});

	async function loadKBs() {
		if (!token && typeof localStorage !== 'undefined') {
			token = localStorage.token || '';
		}
		if (!token) {
			loading = false;
			return;
		}
		try {
			kbs = await getAvailableKBs(token);
		} catch (e) {
			kbs = [];
		}
		loading = false;
	}

	async function ensureSubtags(kbId: number) {
		if (subtagsByKB[kbId] !== undefined) return;
		subtagsLoading = { ...subtagsLoading, [kbId]: true };
		try {
			subtagsByKB = { ...subtagsByKB, [kbId]: await getKBSubtags(token, kbId) };
		} catch (e) {
			subtagsByKB = { ...subtagsByKB, [kbId]: [] };
		}
		subtagsLoading = { ...subtagsLoading, [kbId]: false };
	}

	async function toggleExpand(kbId: number) {
		expanded = { ...expanded, [kbId]: !expanded[kbId] };
		if (expanded[kbId]) await ensureSubtags(kbId);
	}

	function findSel(kbId: number): KBSelection | undefined {
		return selections.find((s) => s.kb_id === kbId);
	}

	function kbState(kbId: number): 'unchecked' | 'whole' | 'scoped' {
		const sel = findSel(kbId);
		if (!sel) return 'unchecked';
		return sel.subtag_ids && sel.subtag_ids.length > 0 ? 'scoped' : 'whole';
	}

	async function persist() {
		if (!chatId) return;
		try {
			await setChatKBConfig(token, chatId, selections);
		} catch (e) {
			console.warn('Failed to save KB config:', e);
		}
	}

	async function toggleWholeKB(kbId: number) {
		const state = kbState(kbId);
		if (state === 'unchecked') {
			selections = [...selections, { kb_id: kbId, subtag_ids: [] }];
		} else {
			// Whole or scoped -> remove entirely
			selections = selections.filter((s) => s.kb_id !== kbId);
		}
		await persist();
	}

	async function toggleSubtag(kbId: number, subtagId: number) {
		const existing = findSel(kbId);
		if (!existing) {
			selections = [...selections, { kb_id: kbId, subtag_ids: [subtagId] }];
		} else {
			const subs = existing.subtag_ids || [];
			const nextSubs = subs.includes(subtagId)
				? subs.filter((id) => id !== subtagId)
				: [...subs, subtagId];
			if (nextSubs.length === 0) {
				// Last subtag unchecked -> remove KB entry entirely
				selections = selections.filter((s) => s.kb_id !== kbId);
			} else {
				selections = selections.map((s) =>
					s.kb_id === kbId ? { ...s, subtag_ids: nextSubs } : s
				);
			}
		}
		await persist();
	}

	function handleClickOutside() {
		if (open) open = false;
	}

	$: label = (() => {
		if (selections.length === 0) return 'KBs';
		const scoped = selections.filter((s) => s.subtag_ids && s.subtag_ids.length > 0);
		const whole = selections.filter((s) => !s.subtag_ids || s.subtag_ids.length === 0);
		if (scoped.length === 0) {
			return `${selections.length} KB${selections.length > 1 ? 's' : ''}`;
		}
		let text: string;
		if (whole.length === 0) {
			text = `${scoped.length} scoped`;
		} else {
			text = `${whole.length} KB + ${scoped.length} scoped`;
		}
		return text.length > 28 ? text.slice(0, 27) + '…' : text;
	})();
</script>

<svelte:window on:click={handleClickOutside} />

{#if token}
	<!-- svelte-ignore a11y-click-events-have-key-events -->
	<!-- svelte-ignore a11y-no-static-element-interactions -->
	<div class="relative inline-block" on:click|stopPropagation>
		<button
			class="flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-lg
				   {locked
						? 'bg-gray-100 dark:bg-gray-900 text-gray-400 dark:text-gray-500 cursor-not-allowed'
						: 'bg-gray-50 dark:bg-gray-850 hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-600 dark:text-gray-300'}
				   transition-colors border border-gray-200 dark:border-gray-700"
			on:click={() => !locked && (open = !open)}
			disabled={locked}
			title={locked
				? 'KB is locked for this chat — start a new chat to change'
				: 'Select Knowledge Bases for this chat'}
		>
			<svg xmlns="http://www.w3.org/2000/svg" class="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none"
				stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
				<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
				<path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
			</svg>
			{label}
			{#if locked}
				<svg xmlns="http://www.w3.org/2000/svg" class="w-3 h-3" viewBox="0 0 24 24"
					fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
					<rect x="3" y="11" width="18" height="11" rx="2" />
					<path d="M7 11V7a5 5 0 0 1 10 0v4" />
				</svg>
			{:else}
				<svg xmlns="http://www.w3.org/2000/svg"
					class="w-3 h-3 transition-transform duration-200 {open ? 'rotate-180' : ''}"
					viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
					<polyline points="6 9 12 15 18 9" />
				</svg>
			{/if}
		</button>

		{#if open}
			<div class="absolute left-0 mt-1.5 w-80 bg-white dark:bg-gray-900 rounded-xl shadow-xl
						border border-gray-200 dark:border-gray-700 z-[999] max-h-80 overflow-y-auto">
				<div class="px-3 py-2 border-b border-gray-100 dark:border-gray-800">
					<span class="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide">
						Knowledge Bases
					</span>
				</div>
				{#if loading}
					<div class="p-4 text-sm text-gray-400 text-center">Loading...</div>
				{:else if kbs.length === 0}
					<div class="p-4 text-sm text-gray-400 text-center">No knowledge bases available</div>
				{:else}
					{#each kbs as kb (kb.id)}
						{@const sel = selections.find((s) => s.kb_id === kb.id)}
						{@const state = !sel
							? 'unchecked'
							: (sel.subtag_ids ?? []).length > 0
								? 'scoped'
								: 'whole'}
						<div class="border-b border-gray-50 dark:border-gray-800/50 last:border-b-0">
							<div class="flex items-center gap-2 px-3 py-2.5 hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors">
								<!-- Expand chevron -->
								<button
									class="w-5 h-5 flex items-center justify-center text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 flex-shrink-0"
									on:click={() => toggleExpand(kb.id)}
									title="Show subtags"
								>
									<svg xmlns="http://www.w3.org/2000/svg"
										class="w-3 h-3 transition-transform duration-150 {expanded[kb.id] ? 'rotate-90' : ''}"
										viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
										stroke-linecap="round" stroke-linejoin="round">
										<polyline points="9 18 15 12 9 6" />
									</svg>
								</button>
								<!-- Tri-state checkbox (whole-KB toggle) -->
								<button
									class="flex-1 flex items-center gap-3 text-left text-sm"
									on:click={() => toggleWholeKB(kb.id)}
								>
									<div class="w-5 h-5 rounded flex-shrink-0 border-2 flex items-center justify-center transition-colors
											{state === 'whole'
												? 'bg-blue-500 border-blue-500'
												: state === 'scoped'
													? 'bg-blue-100 dark:bg-blue-900/40 border-blue-500'
													: 'border-gray-300 dark:border-gray-600'}">
										{#if state === 'whole'}
											<svg class="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24"
												stroke="currentColor" stroke-width="3">
												<polyline points="20 6 9 17 4 12" />
											</svg>
										{:else if state === 'scoped'}
											<div class="w-2 h-2 rounded-sm bg-blue-500"></div>
										{/if}
									</div>
									<div class="flex-1 min-w-0">
										<div class="font-medium text-gray-800 dark:text-gray-200 truncate">{kb.name}</div>
										{#if kb.description}
											<div class="text-xs text-gray-400 dark:text-gray-500 truncate mt-0.5">
												{kb.description}
											</div>
										{/if}
									</div>
								</button>
							</div>

							{#if expanded[kb.id]}
								<div class="pl-10 pr-3 pb-2 bg-gray-50/50 dark:bg-gray-850/30">
									{#if subtagsLoading[kb.id]}
										<div class="py-2 text-xs text-gray-400">Loading subtags…</div>
									{:else if !subtagsByKB[kb.id] || subtagsByKB[kb.id].length === 0}
										<div class="py-2 text-xs text-gray-400 italic">No subtags</div>
									{:else}
										{#each subtagsByKB[kb.id] as st (st.id)}
											{@const checked = (sel?.subtag_ids ?? []).includes(st.id)}
											<button
												class="w-full flex items-center gap-2 py-1.5 text-left text-xs hover:bg-gray-100/50 dark:hover:bg-gray-800/50 rounded px-1"
												on:click={() => toggleSubtag(kb.id, st.id)}
											>
												<div class="w-4 h-4 rounded flex-shrink-0 border-2 flex items-center justify-center transition-colors
														{checked
															? 'bg-blue-500 border-blue-500'
															: 'border-gray-300 dark:border-gray-600'}">
													{#if checked}
														<svg class="w-2.5 h-2.5 text-white" fill="none" viewBox="0 0 24 24"
															stroke="currentColor" stroke-width="3">
															<polyline points="20 6 9 17 4 12" />
														</svg>
													{/if}
												</div>
												<span class="truncate text-gray-700 dark:text-gray-300">{st.name}</span>
											</button>
										{/each}
									{/if}
								</div>
							{/if}
						</div>
					{/each}
				{/if}
			</div>
		{/if}
	</div>
{/if}
