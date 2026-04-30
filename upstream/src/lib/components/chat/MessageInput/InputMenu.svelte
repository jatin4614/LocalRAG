<script lang="ts">
	import { getContext } from 'svelte';
	import { fly } from 'svelte/transition';

	import { user } from '$lib/stores';

	import Dropdown from '$lib/components/common/Dropdown.svelte';
	import Tooltip from '$lib/components/common/Tooltip.svelte';
	import Clip from '$lib/components/icons/Clip.svelte';
	import Photo from '$lib/components/icons/Photo.svelte';

	const i18n = getContext('i18n');

	// Props kept for parent compatibility; only file/image upload is exposed
	// to users in this build. Other attach options (capture, web, knowledge,
	// notes, chats, Google Drive, OneDrive) were removed deliberately as
	// part of the simplified UX.
	export let files = [];

	export let selectedModels: string[] = [];
	export let fileUploadCapableModels: string[] = [];

	export let uploadFilesHandler: Function;
	export let uploadImagesHandler: Function;

	export let onClose: Function;

	let show = false;

	let fileUploadEnabled = true;
	$: fileUploadEnabled =
		fileUploadCapableModels.length === selectedModels.length &&
		($user?.role === 'admin' || $user?.permissions?.chat?.file_upload);

	$: if (!fileUploadEnabled && files.length > 0) {
		files = [];
	}

	const tooltipText = (() => {
		if (fileUploadCapableModels.length !== selectedModels.length) {
			return $i18n.t('Model(s) do not support file upload');
		}
		if (!fileUploadEnabled) {
			return $i18n.t('You do not have permission to upload files.');
		}
		return '';
	});
</script>

<Dropdown
	bind:show
	on:change={(e) => {
		if (e.detail === false) {
			onClose();
		}
	}}
>
	<Tooltip content={$i18n.t('More')}>
		<slot />
	</Tooltip>

	<div slot="content">
		<div
			class="w-56 rounded-2xl px-1 py-1 border border-gray-100 dark:border-gray-800 z-50 bg-white dark:bg-gray-850 dark:text-white shadow-lg transition"
		>
			<div in:fly={{ x: -20, duration: 150 }}>
				<Tooltip content={tooltipText()} className="w-full">
					<button
						class="flex w-full gap-2 items-center px-3 py-1.5 text-sm select-none cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800/50 rounded-xl {!fileUploadEnabled
							? 'opacity-50'
							: ''}"
						type="button"
						on:click={() => {
							if (fileUploadEnabled) {
								show = false;
								uploadFilesHandler();
							}
						}}
					>
						<Clip />
						<div class="line-clamp-1">{$i18n.t('Upload File')}</div>
					</button>
				</Tooltip>

				<Tooltip content={tooltipText()} className="w-full">
					<button
						class="flex w-full gap-2 items-center px-3 py-1.5 text-sm select-none cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800/50 rounded-xl {!fileUploadEnabled
							? 'opacity-50'
							: ''}"
						type="button"
						on:click={() => {
							if (fileUploadEnabled) {
								show = false;
								uploadImagesHandler();
							}
						}}
					>
						<Photo />
						<div class="line-clamp-1">{$i18n.t('Upload Image')}</div>
					</button>
				</Tooltip>
			</div>
		</div>
	</div>
</Dropdown>
