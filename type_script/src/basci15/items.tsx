// src/components/TodoItem.tsx
import React from 'react';
import { Todo } from '../types';

interface Props {
  todo: Todo;
  onDelete: (id: number) => void;
}

const TodoItem: React.FC<Props> = ({ todo, onDelete }) => {
  return (
    <li style={{ marginBottom: '8px' }}>
      <span>{todo.text}</span>
      <button onClick={() => onDelete(todo.id)} style={{ marginLeft: '8px' }}>
        削除
      </button>
    </li>
  );
};

export default TodoItem;