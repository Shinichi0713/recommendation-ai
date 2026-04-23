import { store } from "../store";
import { fetchUserById } from "../api";

// ユーザー一覧画面
export function renderUserList(
  container: HTMLElement,
  navigateTo: (route: `/users/${number}`) => void
): void {
  const state = store.getState();

  const title = document.createElement("h1");
  title.textContent = "ユーザー一覧";

  const loadingDiv = document.createElement("div");
  loadingDiv.textContent = "読み込み中...";
  loadingDiv.style.display = state.loading ? "block" : "none";

  const errorDiv = document.createElement("div");
  errorDiv.style.color = "red";
  errorDiv.textContent = state.error || "";
  errorDiv.style.display = state.error ? "block" : "none";

  const list = document.createElement("ul");
  state.users.forEach(user => {
    const item = document.createElement("li");
    const link = document.createElement("a");
    link.href = `#/users/${user.id}`;
    link.textContent = `${user.name} (${user.email})`;
    link.onclick = e => {
      e.preventDefault();
      navigateTo(`/users/${user.id}`);
    };
    item.appendChild(link);
    list.appendChild(item);
  });

  container.appendChild(title);
  container.appendChild(loadingDiv);
  container.appendChild(errorDiv);
  container.appendChild(list);

  // 状態変更を監視して再描画
  const unsubscribe = store.subscribe(() => {
    // 簡易的に再描画（実際は差分更新が望ましい）
    container.innerHTML = "";
    renderUserList(container, navigateTo);
  });

  // コンポーネント破棄時に購読解除（簡易版）
  (container as any).__unsubscribe = unsubscribe;
}

// ユーザー詳細画面
export function renderUserDetail(
  container: HTMLElement,
  userId: number,
  navigateTo: (route: "/users") => void
): void {
  const state = store.getState();

  const backLink = document.createElement("a");
  backLink.href = "#/users";
  backLink.textContent = "← 一覧に戻る";
  backLink.onclick = e => {
    e.preventDefault();
    navigateTo("/users");
  };

  const title = document.createElement("h1");
  title.textContent = "ユーザー詳細";

  const loadingDiv = document.createElement("div");
  loadingDiv.textContent = "読み込み中...";

  const errorDiv = document.createElement("div");
  errorDiv.style.color = "red";

  const contentDiv = document.createElement("div");

  container.appendChild(backLink);
  container.appendChild(title);
  container.appendChild(loadingDiv);
  container.appendChild(errorDiv);
  container.appendChild(contentDiv);

  // 既にストアにユーザー情報があればそれを使う
  const existingUser = state.users.find(u => u.id === userId);
  if (existingUser) {
    renderUserContent(contentDiv, existingUser);
    loadingDiv.style.display = "none";
  } else {
    // なければAPIから取得
    loadUserDetail(userId, loadingDiv, errorDiv, contentDiv);
  }

  // 簡易的なクリーンアップ（実際はライフサイクル管理が必要）
  (container as any).__cleanup = () => {
    // 必要に応じてタイマーやイベントリスナーを解除
  };
}

async function loadUserDetail(
  userId: number,
  loadingDiv: HTMLElement,
  errorDiv: HTMLElement,
  contentDiv: HTMLElement
): Promise<void> {
  try {
    const user = await fetchUserById(userId);
    if (user) {
      renderUserContent(contentDiv, user);
    } else {
      errorDiv.textContent = "ユーザーが見つかりませんでした";
    }
  } catch (err) {
    errorDiv.textContent = "詳細の取得に失敗しました";
  } finally {
    loadingDiv.style.display = "none";
  }
}

function renderUserContent(container: HTMLElement, user: any): void {
  container.innerHTML = `
    <p><strong>ID:</strong> ${user.id}</p>
    <p><strong>名前:</strong> ${user.name}</p>
    <p><strong>メール:</strong> ${user.email}</p>
    <p><strong>年齢:</strong> ${user.age ?? "未設定"}</p>
  `;
}