const WEBUI_API_BASE_URL = '';

export type KBSelection = { kb_id: number; subtag_ids: number[] };

export const getAvailableKBs = async (token: string) => {
	const res = await fetch(`${WEBUI_API_BASE_URL}/api/kb/available`, {
		headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' }
	});
	if (!res.ok) return [];
	return res.json();
};

export const getKBSubtags = async (
	token: string,
	kbId: number
): Promise<Array<{ id: number; name: string; description: string | null }>> => {
	const res = await fetch(`${WEBUI_API_BASE_URL}/api/kb/${kbId}/subtags`, {
		headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' }
	});
	if (!res.ok) return [];
	return res.json();
};

export const setChatKBConfig = async (token: string, chatId: string, config: any[]) => {
	const res = await fetch(`${WEBUI_API_BASE_URL}/api/chats/${chatId}/kb_config`, {
		method: 'PUT',
		headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
		body: JSON.stringify({ config })
	});
	if (!res.ok) return null;
	return res.json();
};

export const getChatKBConfig = async (token: string, chatId: string) => {
	const res = await fetch(`${WEBUI_API_BASE_URL}/api/chats/${chatId}/kb_config`, {
		headers: { Authorization: `Bearer ${token}` }
	});
	if (!res.ok) return null;
	return res.json();
};

export const uploadKBDoc = async (token: string, kbId: number, subtagId: number, file: File) => {
	const formData = new FormData();
	formData.append('file', file);
	const res = await fetch(`${WEBUI_API_BASE_URL}/api/kb/${kbId}/subtag/${subtagId}/upload`, {
		method: 'POST',
		headers: { Authorization: `Bearer ${token}` },
		body: formData
	});
	if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
	return res.json();
};

export const uploadPrivateDoc = async (token: string, chatId: string, file: File) => {
	const formData = new FormData();
	formData.append('file', file);
	const res = await fetch(`${WEBUI_API_BASE_URL}/api/chats/${chatId}/private_docs/upload`, {
		method: 'POST',
		headers: { Authorization: `Bearer ${token}` },
		body: formData
	});
	if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
	return res.json();
};

export const ragRetrieve = async (
	token: string,
	chatId: string,
	query: string,
	kbConfig: any[]
) => {
	const res = await fetch(`${WEBUI_API_BASE_URL}/api/rag/retrieve`, {
		method: 'POST',
		headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
		body: JSON.stringify({
			chat_id: parseInt(chatId),
			query,
			selected_kb_config: kbConfig
		})
	});
	if (!res.ok) return { hits: [] };
	return res.json();
};
