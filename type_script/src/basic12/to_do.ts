// タスクの型定義
interface Task {
  id: number;
  text: string;
  completed: boolean;
  createdAt: Date;
}

// フィルタの種類
type FilterType = "all" | "active" | "completed";

class TodoApp {
  private tasks: Task[] = [];
  private nextId: number = 1;
  private currentFilter: FilterType = "all";

  private taskInput: HTMLInputElement;
  private addBtn: HTMLButtonElement;
  private taskList: HTMLUListElement;
  private filterAllBtn: HTMLButtonElement;
  private filterActiveBtn: HTMLButtonElement;
  private filterCompletedBtn: HTMLButtonElement;

  constructor() {
    this.taskInput = document.getElementById("taskInput") as HTMLInputElement;
    this.addBtn = document.getElementById("addBtn") as HTMLButtonElement;
    this.taskList = document.getElementById("taskList") as HTMLUListElement;
    this.filterAllBtn = document.getElementById("filterAll") as HTMLButtonElement;
    this.filterActiveBtn = document.getElementById("filterActive") as HTMLButtonElement;
    this.filterCompletedBtn = document.getElementById("filterCompleted") as HTMLButtonElement;

    this.setupEventListeners();
    this.loadTasks();
    this.render();
  }

  private setupEventListeners(): void {
    // タスク追加
    this.addBtn.addEventListener("click", () => this.addTask());
    this.taskInput.addEventListener("keypress", (e) => {
      if (e.key === "Enter") {
        this.addTask();
      }
    });

    // フィルタ変更
    this.filterAllBtn.addEventListener("click", () => this.setFilter("all"));
    this.filterActiveBtn.addEventListener("click", () => this.setFilter("active"));
    this.filterCompletedBtn.addEventListener("click", () => this.setFilter("completed"));

    // キーボードショートカット（Delで完了タスクを一括削除）
    document.addEventListener("keydown", (e) => {
      if (e.key === "Delete" && e.ctrlKey) {
        e.preventDefault();
        this.deleteCompletedTasks();
      }
    });
  }

  private addTask(): void {
    const text = this.taskInput.value.trim();
    if (text === "") return;

    const task: Task = {
      id: this.nextId++,
      text,
      completed: false,
      createdAt: new Date(),
    };

    this.tasks.push(task);
    this.taskInput.value = "";
    this.saveTasks();
    this.render();
  }

  private toggleTask(id: number): void {
    const task = this.tasks.find(t => t.id === id);
    if (task) {
      task.completed = !task.completed;
      this.saveTasks();
      this.render();
    }
  }

  private deleteTask(id: number): void {
    this.tasks = this.tasks.filter(t => t.id !== id);
    this.saveTasks();
    this.render();
  }

  private deleteCompletedTasks(): void {
    this.tasks = this.tasks.filter(t => !t.completed);
    this.saveTasks();
    this.render();
  }

  private setFilter(filter: FilterType): void {
    this.currentFilter = filter;

    // フィルタボタンの見た目更新
    [this.filterAllBtn, this.filterActiveBtn, this.filterCompletedBtn].forEach(btn => {
      btn.classList.remove("active");
    });
    document.getElementById(`filter${filter.charAt(0).toUpperCase() + filter.slice(1)}`)?.classList.add("active");

    this.render();
  }

  private getFilteredTasks(): Task[] {
    switch (this.currentFilter) {
      case "active":
        return this.tasks.filter(t => !t.completed);
      case "completed":
        return this.tasks.filter(t => t.completed);
      default:
        return this.tasks;
    }
  }

  private render(): void {
    const filteredTasks = this.getFilteredTasks();

    this.taskList.innerHTML = "";

    if (filteredTasks.length === 0) {
      const emptyMsg = document.createElement("li");
      emptyMsg.textContent = this.currentFilter === "all" 
        ? "タスクがありません" 
        : "該当するタスクがありません";
      emptyMsg.style.textAlign = "center";
      emptyMsg.style.color = "#999";
      this.taskList.appendChild(emptyMsg);
      return;
    }

    filteredTasks.forEach(task => {
      const li = document.createElement("li");
      li.className = "task-item" + (task.completed ? " completed" : "");

      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = task.completed;
      checkbox.addEventListener("change", () => this.toggleTask(task.id));

      const textSpan = document.createElement("span");
      textSpan.className = "task-text";
      textSpan.textContent = task.text;

      const deleteBtn = document.createElement("button");
      deleteBtn.className = "delete-btn";
      deleteBtn.textContent = "削除";
      deleteBtn.addEventListener("click", () => this.deleteTask(task.id));

      li.appendChild(checkbox);
      li.appendChild(textSpan);
      li.appendChild(deleteBtn);

      this.taskList.appendChild(li);
    });
  }

  // ローカルストレージへの保存・復元
  private saveTasks(): void {
    const data = JSON.stringify({
      tasks: this.tasks,
      nextId: this.nextId,
    });
    localStorage.setItem("todoAppData", data);
  }

  private loadTasks(): void {
    const data = localStorage.getItem("todoAppData");
    if (!data) return;

    try {
      const parsed = JSON.parse(data);
      this.tasks = parsed.tasks || [];
      this.nextId = parsed.nextId || 1;
    } catch (err) {
      console.error("タスクデータの読み込みに失敗しました", err);
    }
  }
}

// アプリ起動
document.addEventListener("DOMContentLoaded", () => {
  new TodoApp();
});