<script lang="ts">
	import { getContext, onMount } from 'svelte';
	import { toast } from 'svelte-sonner';

	import {
		WEBUI_NAME,
		banners,
		chatId,
		config,
		mobile,
		selectedKBConfig,
		settings,
		showArchivedChats,
		showControls,
		showSidebar,
		temporaryChatEnabled,
		user
	} from '$lib/stores';

	import { slide } from 'svelte/transition';
	import { page } from '$app/stores';
	import { goto } from '$app/navigation';

	import ShareChatModal from '../chat/ShareChatModal.svelte';
	import KBPickerModal from '../chat/KBPickerModal.svelte';
	import {
		getChatKBConfig,
		setChatKBConfig,
		type KBSelection
	} from '$lib/apis/kb';
	import Tooltip from '../common/Tooltip.svelte';
	import Menu from '$lib/components/layout/Navbar/Menu.svelte';
	import UserMenu from '$lib/components/layout/Sidebar/UserMenu.svelte';
	import AdjustmentsHorizontal from '../icons/AdjustmentsHorizontal.svelte';

	import PencilSquare from '../icons/PencilSquare.svelte';
	import Banner from '../common/Banner.svelte';
	import Sidebar from '../icons/Sidebar.svelte';

	import ChatBubbleDotted from '../icons/ChatBubbleDotted.svelte';
	import ChatBubbleDottedChecked from '../icons/ChatBubbleDottedChecked.svelte';

	import EllipsisHorizontal from '../icons/EllipsisHorizontal.svelte';
	import ChatPlus from '../icons/ChatPlus.svelte';
	import ChatCheck from '../icons/ChatCheck.svelte';
	import Knobs from '../icons/Knobs.svelte';
	import { WEBUI_API_BASE_URL } from '$lib/constants';

	const i18n = getContext('i18n');

	export let initNewChat: Function;
	export let shareEnabled: boolean = false;
	export let scrollTop = 0;

	export let chat;
	export let history;
	export let selectedModels;
	export let showModelSelector = true;

	export let onSaveTempChat: () => {};
	export let archiveChatHandler: (id: string) => void;
	export let moveChatHandler: (id: string, folderId: string) => void;

	let closedBannerIds = [];

	let showShareChatModal = false;

	// KB selection modal. Loads existing config when chatId resolves; persists
	// via PUT /api/chats/{id}/kb_config on confirm. For brand-new chats (no
	// chatId yet) we hold the selection in a pending var and apply it as
	// soon as the chat row exists. Once a chat has its first user message,
	// the backend rejects further kb_config writes (lock invariant), so the
	// frontend mirrors that lock by disabling the picker button below.
	let showKBPicker = false;
	let kbSelections: KBSelection[] = [];
	let pendingKBSelections: KBSelection[] | null = null;
	let lastLoadedChatId: string = '';
	let initialMountDone = false;
	let prevChatIdSeen: string = '';

	$: chatStarted = !!$chatId;

	$: if ($chatId && $chatId !== lastLoadedChatId) {
		lastLoadedChatId = $chatId;
		loadKBConfigForCurrentChat();
	}

	// Auto-open the KB picker whenever we land on a brand-new chat — either
	// on initial mount with no chatId, or on a transition from existing
	// chatId → empty (e.g. user clicked "New Chat"). Once chatStarted, the
	// picker is locked, so we never auto-open it then.
	$: {
		const current = $chatId || '';
		if (initialMountDone && current === '' && prevChatIdSeen !== '') {
			showKBPicker = true;
			kbSelections = [];
			pendingKBSelections = null;
			selectedKBConfig.set([]);
		}
		prevChatIdSeen = current;
	}

	onMount(() => {
		// Defer one tick so $chatId has resolved if we're loading an existing chat.
		setTimeout(() => {
			if (!$chatId) {
				showKBPicker = true;
			}
			initialMountDone = true;
		}, 50);
	});

	async function loadKBConfigForCurrentChat() {
		const token = localStorage.token || '';
		if (!token || !$chatId) return;
		try {
			const data = await getChatKBConfig(token, $chatId);
			const cfg = data?.config ?? data?.selectedKbConfig ?? [];
			const fromDb = (Array.isArray(cfg) ? cfg : []).map((s: any) => ({
				kb_id: s.kb_id,
				subtag_ids: Array.isArray(s.subtag_ids) ? s.subtag_ids : []
			}));
			// If the picker was used before the chat existed, prefer the user's
			// pending choice over an empty DB row. Otherwise hydrate from DB so
			// returning to an existing chat shows its locked-in selection.
			if (pendingKBSelections && pendingKBSelections.length > 0) {
				kbSelections = pendingKBSelections;
				selectedKBConfig.set(pendingKBSelections);
			} else {
				kbSelections = fromDb;
				selectedKBConfig.set(fromDb);
			}
		} catch (_e) {
			kbSelections = [];
			selectedKBConfig.set([]);
		}
		// Try to persist the pending choice to the DB (best-effort — the
		// backend lock will reject this with 409 once the chat row already
		// has a user message; that's fine because retrieval rides on the
		// in-flight kb_config from the chat-completion body, not the DB).
		if (pendingKBSelections && pendingKBSelections.length > 0) {
			const flush = pendingKBSelections;
			pendingKBSelections = null;
			try {
				await persistKBSelections(flush);
			} catch (_e2) {
				// Silently ignored — user-facing toast for 409 would be noise
				// since retrieval still works via the body field.
			}
		}
	}

	async function persistKBSelections(selections: KBSelection[]) {
		const token = localStorage.token || '';
		if (!token || !$chatId) return;
		try {
			const result = await setChatKBConfig(token, $chatId, selections);
			if (result) {
				kbSelections = selections;
				selectedKBConfig.set(selections);
				toast.success(
					selections.length > 0
						? $i18n.t(`Attached ${selections.length} KB(s) to chat`)
						: $i18n.t('Cleared KB selection')
				);
			} else {
				toast.error($i18n.t('Failed to save KB selection'));
			}
		} catch (_e) {
			toast.error($i18n.t('Failed to save KB selection'));
		}
	}

	async function onKBPickerConfirm(
		event: CustomEvent<{ selections: KBSelection[] }>
	) {
		showKBPicker = false;
		const next = event.detail.selections || [];
		// Always update the live store so Chat.svelte sees this on the very
		// next message, regardless of whether the chat row exists yet.
		selectedKBConfig.set(next);
		kbSelections = next;
		if ($chatId) {
			await persistKBSelections(next);
		} else {
			// Chat row doesn't exist yet — durability via PUT happens after
			// the first message lands and chatId resolves. The in-flight
			// kb_config from the body still drives retrieval today.
			pendingKBSelections = next;
		}
	}
	let showDownloadChatModal = false;
</script>

<ShareChatModal bind:show={showShareChatModal} chatId={$chatId} />
<KBPickerModal
	bind:show={showKBPicker}
	selections={kbSelections}
	on:confirm={onKBPickerConfirm}
/>

<button
	id="new-chat-button"
	class="hidden"
	on:click={() => {
		initNewChat();
	}}
	aria-label="New Chat"
/>

<nav
	class="sticky top-0 z-30 w-full {chat?.id
		? 'pt-0.5 pb-1'
		: 'pt-1 pb-1'} -mb-12 flex flex-col items-center drag-region"
>
	<div class="flex items-center w-full pl-1.5 pr-1">
		<div
			id="navbar-bg-gradient-to-b"
			class="{chat?.id
				? 'visible'
				: 'invisible'} bg-linear-to-b via-40% to-97% from-white/90 via-white/50 to-transparent dark:from-gray-900/90 dark:via-gray-900/50 dark:to-transparent pointer-events-none absolute inset-0 -bottom-10 z-[-1]"
		></div>

		<div class=" flex max-w-full w-full mx-auto px-1.5 md:px-2 pt-0.5 bg-transparent">
			<div class="flex items-center w-full max-w-full">
				{#if $mobile && !$showSidebar}
					<div
						class="-translate-x-0.5 mr-1 mt-1 self-start flex flex-none items-center text-gray-600 dark:text-gray-400"
					>
						<Tooltip content={$showSidebar ? $i18n.t('Close Sidebar') : $i18n.t('Open Sidebar')}>
							<button
								class=" cursor-pointer flex rounded-lg hover:bg-gray-100 dark:hover:bg-gray-850 transition"
								on:click={() => {
									showSidebar.set(!$showSidebar);
								}}
							>
								<div class=" self-center p-1.5">
									<Sidebar />
								</div>
							</button>
						</Tooltip>
					</div>
				{/if}

				<div
					class="flex-1 overflow-hidden max-w-full mt-0.5 py-0.5
			{$showSidebar ? 'ml-1' : ''}
			"
				>
					<!-- Model selector removed: this deployment locks all chats to the
						 single base model configured by DEFAULT_MODELS. -->
				</div>

				<div class="self-start flex flex-none items-center text-gray-600 dark:text-gray-400">
					<!-- <div class="md:hidden flex self-center w-[1px] h-5 mx-2 bg-gray-300 dark:bg-stone-700" /> -->

					<!-- KB selector modal trigger. While the chat hasn't started yet
						 (no chatId), the button is interactive and the picker also
						 auto-opens once on entry. Once a chat has started, the button
						 becomes a non-interactive locked badge — the backend would
						 reject any kb_config change at this point. -->
					<Tooltip
						content={chatStarted
							? $i18n.t('Knowledge Base is locked once the chat has started')
							: $i18n.t('Select Knowledge Base')}
					>
						<button
							class="flex items-center px-2 py-2 rounded-xl transition {chatStarted
								? 'cursor-not-allowed opacity-70'
								: 'cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-850'}"
							id="kb-picker-button"
							aria-label={chatStarted
								? $i18n.t('Knowledge Base (locked)')
								: $i18n.t('Select Knowledge Base')}
							disabled={chatStarted}
							on:click={() => {
								if (!chatStarted) showKBPicker = true;
							}}
						>
							<svg
								xmlns="http://www.w3.org/2000/svg"
								fill="none"
								viewBox="0 0 24 24"
								stroke-width="1.5"
								stroke="currentColor"
								class="size-5"
							>
								<path
									stroke-linecap="round"
									stroke-linejoin="round"
									d="M12 6.042A8.967 8.967 0 0 0 6 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 0 1 6 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 0 1 6-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0 0 18 18a8.967 8.967 0 0 0-6 2.292m0-14.25v14.25"
								/>
							</svg>
							{#if kbSelections.length > 0}
								<span class="ml-1 text-xs font-semibold text-blue-600 dark:text-blue-400">
									{kbSelections.length}
								</span>
							{/if}
							{#if chatStarted}
								<svg
									xmlns="http://www.w3.org/2000/svg"
									viewBox="0 0 20 20"
									fill="currentColor"
									class="ml-0.5 size-3.5 text-gray-400"
								>
									<path
										fill-rule="evenodd"
										d="M10 1a4.5 4.5 0 0 0-4.5 4.5V9H5a2 2 0 0 0-2 2v6a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-6a2 2 0 0 0-2-2h-.5V5.5A4.5 4.5 0 0 0 10 1Zm3 8V5.5a3 3 0 1 0-6 0V9h6Z"
										clip-rule="evenodd"
									/>
								</svg>
							{/if}
						</button>
					</Tooltip>

					{#if $user?.role === 'user' ? ($user?.permissions?.chat?.temporary ?? true) && !($user?.permissions?.chat?.temporary_enforced ?? false) : true}
						{#if !chat?.id}
							<Tooltip content={$i18n.t(`Temporary Chat`)}>
								<button
									class="flex cursor-pointer px-2 py-2 rounded-xl hover:bg-gray-50 dark:hover:bg-gray-850 transition"
									id="temporary-chat-button"
									on:click={async () => {
										if (($settings?.temporaryChatByDefault ?? false) && $temporaryChatEnabled) {
											// for proper initNewChat handling
											await temporaryChatEnabled.set(null);
										} else {
											await temporaryChatEnabled.set(!$temporaryChatEnabled);
										}

										if ($page.url.pathname !== '/') {
											await goto('/');
										}

										// add 'temporary-chat=true' to the URL
										if ($temporaryChatEnabled) {
											window.history.replaceState(null, '', '?temporary-chat=true');
										} else {
											window.history.replaceState(null, '', location.pathname);
										}
									}}
								>
									<div class=" m-auto self-center">
										{#if $temporaryChatEnabled}
											<ChatBubbleDottedChecked className=" size-4.5" strokeWidth="1.5" />
										{:else}
											<ChatBubbleDotted className=" size-4.5" strokeWidth="1.5" />
										{/if}
									</div>
								</button>
							</Tooltip>
						{:else if $temporaryChatEnabled}
							<Tooltip content={$i18n.t(`Save Chat`)}>
								<button
									class="flex cursor-pointer px-2 py-2 rounded-xl hover:bg-gray-50 dark:hover:bg-gray-850 transition"
									id="save-temporary-chat-button"
									on:click={async () => {
										onSaveTempChat();
									}}
								>
									<div class=" m-auto self-center">
										<ChatCheck className=" size-4.5" strokeWidth="1.5" />
									</div>
								</button>
							</Tooltip>
						{/if}
					{/if}

					{#if $mobile && !$temporaryChatEnabled && chat && chat.id}
						<Tooltip content={$i18n.t('New Chat')}>
							<button
								class=" flex {$showSidebar
									? 'md:hidden'
									: ''} cursor-pointer px-2 py-2 rounded-xl text-gray-600 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-gray-850 transition"
								on:click={() => {
									initNewChat();
								}}
								aria-label="New Chat"
							>
								<div class=" m-auto self-center">
									<ChatPlus className=" size-4.5" strokeWidth="1.5" />
								</div>
							</button>
						</Tooltip>
					{/if}

					{#if shareEnabled && chat && (chat.id || $temporaryChatEnabled)}
						<Menu
							{chat}
							{shareEnabled}
							shareHandler={() => {
								showShareChatModal = !showShareChatModal;
							}}
							archiveChatHandler={() => {
								archiveChatHandler(chat.id);
							}}
							{moveChatHandler}
						>
							<button
								class="flex cursor-pointer px-2 py-2 rounded-xl hover:bg-gray-50 dark:hover:bg-gray-850 transition"
								id="chat-context-menu-button"
							>
								<div class=" m-auto self-center">
									<EllipsisHorizontal className=" size-5" strokeWidth="1.5" />
								</div>
							</button>
						</Menu>
					{/if}

					{#if $user?.role === 'admin' || ($user?.permissions.chat?.controls ?? true)}
						<Tooltip content={$i18n.t('Controls')}>
							<button
								class=" flex cursor-pointer px-2 py-2 rounded-xl hover:bg-gray-50 dark:hover:bg-gray-850 transition"
								on:click={async () => {
									await showControls.set(!$showControls);
								}}
								aria-label="Controls"
							>
								<div class=" m-auto self-center">
									<Knobs className=" size-5" strokeWidth="1" />
								</div>
							</button>
						</Tooltip>
					{/if}

					{#if $user !== undefined && $user !== null}
						<UserMenu
							className="w-[240px]"
							role={$user?.role}
							help={true}
							on:show={(e) => {
								if (e.detail === 'archived-chat') {
									showArchivedChats.set(true);
								}
							}}
						>
							<div
								class="select-none flex rounded-xl p-1.5 w-full hover:bg-gray-50 dark:hover:bg-gray-850 transition"
							>
								<div class=" self-center">
									<span class="sr-only">{$i18n.t('User menu')}</span>
									<img
										src={`${WEBUI_API_BASE_URL}/users/${$user?.id}/profile/image`}
										class="size-6 object-cover rounded-full"
										alt=""
										draggable="false"
									/>
								</div>
							</div>
						</UserMenu>
					{/if}
				</div>
			</div>
		</div>
	</div>

	{#if $temporaryChatEnabled && ($chatId ?? '').startsWith('local:')}
		<div class=" w-full z-30 text-center">
			<div class="text-xs text-gray-500">{$i18n.t('Temporary Chat')}</div>
		</div>
	{/if}

	<div class="absolute top-[100%] left-0 right-0 h-fit">
		{#if !history.currentId && !$chatId && ($banners.length > 0 || ($config?.license_metadata?.type ?? null) === 'trial' || (($config?.license_metadata?.seats ?? null) !== null && $config?.user_count > $config?.license_metadata?.seats))}
			<div class=" w-full z-30">
				<div class=" flex flex-col gap-1 w-full">
					{#if ($config?.license_metadata?.type ?? null) === 'trial'}
						<Banner
							banner={{
								type: 'info',
								title: 'Trial License',
								content: $i18n.t(
									'You are currently using a trial license. Please contact support to upgrade your license.'
								)
							}}
						/>
					{/if}

					{#if ($config?.license_metadata?.seats ?? null) !== null && $config?.user_count > $config?.license_metadata?.seats}
						<Banner
							banner={{
								type: 'error',
								title: 'License Error',
								content: $i18n.t(
									'Exceeded the number of seats in your license. Please contact support to increase the number of seats.'
								)
							}}
						/>
					{/if}

					{#each $banners.filter((b) => ![...JSON.parse(localStorage.getItem('dismissedBannerIds') ?? '[]'), ...closedBannerIds].includes(b.id)) as banner (banner.id)}
						<Banner
							{banner}
							on:dismiss={(e) => {
								const bannerId = e.detail;

								if (banner.dismissible) {
									localStorage.setItem(
										'dismissedBannerIds',
										JSON.stringify(
											[
												bannerId,
												...JSON.parse(localStorage.getItem('dismissedBannerIds') ?? '[]')
											].filter((id) => $banners.find((b) => b.id === id))
										)
									);
								} else {
									closedBannerIds = [...closedBannerIds, bannerId];
								}
							}}
						/>
					{/each}
				</div>
			</div>
		{/if}
	</div>
</nav>
