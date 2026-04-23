import { store } from "./store";
import { fetchUsers, fetchUserById } from "./api";
import { renderUserList, renderUserDetail } from "./components/userViews";

// ルート定義
type Route = "/users" | `/users/${number}`;

// 現在のルートを管理
let currentRoute: Route = "/users";

// ルーティング関数
function navigateTo(route: Route): void {
  currentRoute = route;
  render();
}

// URLハッシュからルートを取得（簡易版）
function getRouteFromHash(): Route {
  const hash = window.location.hash.slice(1) || "/users";
  if (hash.startsWith("/users/")) {
    const id = parseInt(hash.split("/")[2], 10);
    if (!isNaN(id)) {
      return `/users/${id}` as Route;
    }
  }
  return "/users";
}

// ハッシュ変更イベントを監視
window.addEventListener("hashchange", () => {
  const route = getRouteFromHash();
  navigateTo(route);
});

// メインのレンダリング関数
function render(): void {
  const appElement = document.getElementById("app")!;
  appElement.innerHTML = "";

  if (currentRoute === "/users") {
    renderUserList(appElement, navigateTo);
  } else if (currentRoute.startsWith("/users/")) {
    const id = parseInt(currentRoute.split("/")[2], 10);
    renderUserDetail(appElement, id, navigateTo);
  }
}

// 初期化処理
async function init(): Promise<void> {
  store.dispatch({ type: "SET_LOADING", payload: true });
  try {
    const users = await fetchUsers();
    store.dispatch({ type: "SET_USERS", payload: users });
  } catch (err) {
    store.dispatch({
      type: "SET_ERROR",
      payload: "ユーザー一覧の取得に失敗しました",
    });
  } finally {
    store.dispatch({ type: "SET_LOADING", payload: false });
  }

  // 初期ルートを設定してレンダリング
  currentRoute = getRouteFromHash();
  render();
}

// アプリ起動
init();