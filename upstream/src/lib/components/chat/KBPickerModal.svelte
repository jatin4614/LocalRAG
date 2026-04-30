<script lang="ts">
	import { onMount } from 'svelte';
	import { createEventDispatcher } from 'svelte';
	import { getAvailableKBs, getKBSubtags, type KBSelection } from '$lib/apis/kb';

	export let show = false;
	export let selections: KBSelection[] = [];

	const dispatch = createEventDispatcher<{ confirm: { selections: KBSelection[] } }>();

	let kbs: any[] = [];
	let loading = true;

	// Per-KB UI state
	let expanded: Record<number, boolean> = {};
	let subtagsByKB: Record<number, any[]> = {};
	let subtagLoading: Record<number, boolean> = {};
	let subtagFetched: Record<number, boolean> = {};

	onMount(async () => {
		await loadKBs();
	});

	function getToken(): string {
		return typeof localStorage !== 'undefined' ? (localStorage.token || '') : '';
	}

	async function loadKBs() {
		const token = getToken();
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

	async function ensureSubtagsLoaded(kbId: number) {
		if (subtagFetched[kbId]) return;
		subtagLoading[kbId] = true;
		subtagLoading = { ...subtagLoading };
		const token = getToken();
		try {
			const result = await getKBSubtags(token, kbId);
			subtagsByKB[kbId] = Array.isArray(result) ? result : [];
		} catch (e) {
			// Fail-open: show "No subtags" instead of erroring the modal.
			subtagsByKB[kbId] = [];
		}
		subtagsByKB = { ...subtagsByKB };
		subtagLoading[kbId] = false;
		subtagLoading = { ...subtagLoading };
		subtagFetched[kbId] = true;
		subtagFetched = { ...subtagFetched };
	}

	function findEntry(kbId: number): KBSelection | undefined {
		return selections.find((s) => s.kb_id === kbId);
	}

	function rowState(kbId: number): 'unchecked' | 'whole' | 'partial' {
		const entry = findEntry(kbId);
		if (!entry) return 'unchecked';
		if (entry.subtag_ids.length === 0) return 'whole';
		return 'partial';
	}

	async function toggleExpand(kbId: number) {
		expanded[kbId] = !expanded[kbId];
		expanded = { ...expanded };
		if (expanded[kbId]) {
			await ensureSubtagsLoaded(kbId);
		}
	}

	function toggleKBCheckbox(kbId: number) {
		const state = rowState(kbId);
		if (state === 'unchecked') {
			// Add whole-KB entry.
			selections = [...selections, { kb_id: kbId, subtag_ids: [] }];
		} else if (state === 'whole') {
			// Remove entry entirely.
			selections = selections.filter((s) => s.kb_id !== kbId);
		} else {
			// partial → whole: clear subtag_ids.
			selections = selections.map((s) =>
				s.kb_id === kbId ? { kb_id: kbId, subtag_ids: [] } : s
			);
		}
	}

	function toggleSubtag(kbId: number, subtagId: number) {
		const entry = findEntry(kbId);
		if (!entry) {
			// First subtag pick on previously unchecked KB.
			selections = [...selections, { kb_id: kbId, subtag_ids: [subtagId] }];
			return;
		}
		if (entry.subtag_ids.length === 0) {
			// Currently whole-KB; a subtag click narrows to that one subtag.
			selections = selections.map((s) =>
				s.kb_id === kbId ? { kb_id: kbId, subtag_ids: [subtagId] } : s
			);
			return;
		}
		if (entry.subtag_ids.includes(subtagId)) {
			const next = entry.subtag_ids.filter((x) => x !== subtagId);
			if (next.length === 0) {
				// Removed last subtag → remove entry entirely (collapses to unchecked).
				selections = selections.filter((s) => s.kb_id !== kbId);
			} else {
				selections = selections.map((s) =>
					s.kb_id === kbId ? { kb_id: kbId, subtag_ids: next } : s
				);
			}
		} else {
			selections = selections.map((s) =>
				s.kb_id === kbId ? { kb_id: kbId, subtag_ids: [...s.subtag_ids, subtagId] } : s
			);
		}
	}

	function confirmWithSelections() {
		dispatch('confirm', { selections });
		show = false;
	}

	function confirmWithoutKBs() {
		selections = [];
		dispatch('confirm', { selections: [] });
		show = false;
	}

	$: wholeKBCount = selections.filter((s) => s.subtag_ids.length === 0).length;
	$: subtagCount = selections.reduce((acc, s) => acc + s.subtag_ids.length, 0);
	$: hasSelection = selections.length > 0;
	$: buttonLabel = (() => {
		if (!hasSelection) return 'Use selection';
		const parts: string[] = [];
		if (wholeKBCount > 0) parts.push(`${wholeKBCount} KB`);
		if (subtagCount > 0) parts.push(`${subtagCount} subtag`);
		return `Use ${parts.join(' + ')} selection(s)`;
	})();
</script>

{#if show}
	<div class="fixed inset-0 z-[9999] flex items-center justify-center bg-black/50 backdrop-blur-sm">
		<div class="bg-white dark:bg-gray-900 rounded-2xl shadow-2xl max-w-lg w-full mx-4 p-6 border border-gray-200 dark:border-gray-800">
			<div class="mb-5">
				<h2 class="text-lg font-semibold text-gray-900 dark:text-gray-100">
					Choose Knowledge Bases
				</h2>
				<p class="text-xs text-gray-500 dark:text-gray-400 mt-1">
					Select which knowledge bases this conversation should reference. You cannot change this after sending the first message.
				</p>
			</div>

			<div class="max-h-72 overflow-y-auto mb-4 border border-gray-100 dark:border-gray-800 rounded-xl divide-y divide-gray-100 dark:divide-gray-800">
				{#if loading}
					<div class="p-4 text-sm text-gray-400 text-center">Loading…</div>
				{:else if kbs.length === 0}
					<div class="p-6 text-sm text-gray-400 text-center">
						No knowledge bases available to you. You can continue without a KB.
					</div>
				{:else}
					{#each kbs as kb (kb.id)}
						{@const entry = selections.find((s) => s.kb_id === kb.id)}
						{@const state = !entry
							? 'unchecked'
							: entry.subtag_ids.length === 0
								? 'whole'
								: 'partial'}
						<div class="select-none">
							<!-- svelte-ignore a11y-click-events-have-key-events -->
							<!-- svelte-ignore a11y-no-static-element-interactions -->
							<div
								class="flex items-center gap-3 p-3 hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors"
							>
								<!-- svelte-ignore a11y-click-events-have-key-events -->
								<!-- svelte-ignore a11y-no-static-element-interactions -->
								<div
									class="w-5 h-5 rounded flex-shrink-0 border-2 flex items-center justify-center cursor-pointer
											{state === 'whole'
												? 'bg-blue-500 border-blue-500'
												: state === 'partial'
													? 'bg-blue-100 dark:bg-blue-900/40 border-blue-500'
													: 'border-gray-300 dark:border-gray-600'}"
									on:click|stopPropagation={() => toggleKBCheckbox(kb.id)}
								>
									{#if state === 'whole'}
										<svg class="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="3">
											<polyline points="20 6 9 17 4 12" />
										</svg>
									{:else if state === 'partial'}
										<div class="w-2.5 h-0.5 bg-blue-500 rounded"></div>
									{/if}
								</div>
								<!-- svelte-ignore a11y-click-events-have-key-events -->
								<!-- svelte-ignore a11y-no-static-element-interactions -->
								<div
									class="flex-1 min-w-0 cursor-pointer"
									on:click={() => toggleExpand(kb.id)}
								>
									<div class="font-medium text-sm text-gray-800 dark:text-gray-100 truncate">
										{kb.name}
									</div>
									{#if kb.description}
										<div class="text-xs text-gray-400 dark:text-gray-500 truncate mt-0.5">
											{kb.description}
										</div>
									{/if}
								</div>
								<!-- svelte-ignore a11y-click-events-have-key-events -->
								<!-- svelte-ignore a11y-no-static-element-interactions -->
								<button
									type="button"
									class="w-6 h-6 flex items-center justify-center text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 transition-transform"
									class:rotate-90={expanded[kb.id]}
									on:click|stopPropagation={() => toggleExpand(kb.id)}
									aria-label="Expand subtags"
								>
									<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
										<polyline points="9 6 15 12 9 18" />
									</svg>
								</button>
							</div>

							{#if expanded[kb.id]}
								<div class="pl-10 pr-3 pb-3 bg-gray-50/60 dark:bg-gray-800/30">
									{#if subtagLoading[kb.id]}
										<div class="py-2 text-xs text-gray-400">Loading subtags…</div>
									{:else if !subtagsByKB[kb.id] || subtagsByKB[kb.id].length === 0}
										<div class="py-2 text-xs text-gray-400">No subtags</div>
									{:else}
										{#each subtagsByKB[kb.id] as subtag (subtag.id)}
											{@const subtagChecked =
												state === 'whole' ||
												(entry?.subtag_ids.includes(subtag.id) ?? false)}
											<!-- svelte-ignore a11y-click-events-have-key-events -->
											<!-- svelte-ignore a11y-no-static-element-interactions -->
											<div
												class="flex items-center gap-3 py-1.5 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800/50 rounded px-2 -mx-2"
												on:click={() => toggleSubtag(kb.id, subtag.id)}
											>
												<div
													class="w-4 h-4 rounded flex-shrink-0 border-2 flex items-center justify-center
															{subtagChecked
																? 'bg-blue-500 border-blue-500'
																: 'border-gray-300 dark:border-gray-600'}"
												>
													{#if subtagChecked}
														<svg class="w-2.5 h-2.5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="3">
															<polyline points="20 6 9 17 4 12" />
														</svg>
													{/if}
												</div>
												<div class="text-xs text-gray-700 dark:text-gray-200 truncate">
													{subtag.name}
												</div>
											</div>
										{/each}
									{/if}
								</div>
							{/if}
						</div>
					{/each}
				{/if}
			</div>

			<div class="flex items-center justify-end gap-2">
				<button
					class="px-4 py-2 text-sm font-medium rounded-lg text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 transition"
					on:click={confirmWithoutKBs}
				>
					Continue without KB
				</button>
				<button
					class="px-4 py-2 text-sm font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition disabled:opacity-50"
					on:click={confirmWithSelections}
					disabled={!hasSelection}
				>
					{buttonLabel}
				</button>
			</div>
		</div>
	</div>
{/if}
