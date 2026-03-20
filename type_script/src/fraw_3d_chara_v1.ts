import React, { useRef, useState, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Environment, useGLTF } from '@react-three/drei';
import * as THREE from 'three';

// --- (1) かわいいキャラクターコンポーネント ---
const CuteCharacter: React.FC = () => {
    // 💡 パパへのアドバイス：
    // ここで無料の3Dモデル（.glbファイル）を使用します。
    // 例：'bear.glb'というかわいいクマのモデルがある想定です。
    // モデルファイルはpublicフォルダに置いてください。
    const { scene, animations } = useGLTF('/bear.glb'); // ←ここに使用したいモデルのパスを指定
    const group = useRef<THREE.Group>(null);
    const mixer = useRef<THREE.AnimationMixer>();

    // モデルがロードされたらアニメーションを準備
    useEffect(() => {
        if (animations.length) {
            mixer.current = new THREE.AnimationMixer(scene);
            const action = mixer.current.clipAction(animations[0]); // 最初の多くの場合「走り」や「ジャンプ」
            action.play();
        }
    }, [scene, animations]);

    // 💡 AIエンジニア視点：
    // 毎フレーム（useFrame）キャラクターをグラフの周りに移動させ、
    // アニメーションを更新します。
    useFrame((state, delta) => {
        mixer.current?.update(delta); // アニメーションの更新
        if (group.current) {
            const time = state.clock.getElapsedTime();
            // グラフの周りを円を描くように走らせる（x=cos, z=sin）
            const radius = 5;
            group.current.position.x = Math.cos(time) * radius;
            group.current.position.z = Math.sin(time) * radius;
            // 進行方向を向かせる
            group.current.rotation.y = time + Math.PI / 2;
        }
    });

    return <primitive ref={group} object={scene} scale={1.5} />;
};

// --- (2) 3D棒グラフコンポーネント ---
interface GraphBarProps {
    data: number;
    index: number;
}

const GraphBar: React.FC<GraphBarProps> = ({ data, index }) => {
    const meshRef = useRef<THREE.Mesh>(null);

    // AI的なデータ（例：お子様の歯磨き時間）を棒の高さにする
    const height = data / 10; // 高さを調整
    const positionY = height / 2; // 棒の底辺をy=0に合わせる

    return (
        <mesh ref={meshRef} position={[index * 2 - 4, positionY, 0]}>
            <boxGeometry args={[1, height, 1]} />
            <meshStandardMaterial color={new THREE.Color(`hsl(${index * 40}, 80%, 60%)`)} />
        </mesh>
    );
};

// --- (3) メインシーンコンポーネント ---
const CuteGraphScene: React.FC = () => {
    // お子様の歯磨き「頑張りデータ」（例：月曜〜日曜の秒数）
    const dentalLogData = [120, 180, 90, 210, 150, 240, 200];

    return (
        <div style={{ height: '100vh', width: '100vw' }}>
            <h2>今週の歯磨きがんばりグラフ！✨🐻</h2>
            <Canvas shadows camera={{ position: [0, 8, 15], fov: 60 }}>
                {/* 💡 パパへのアドバイス：照明設定でかわいさを出す */}
                <ambientLight intensity={0.5} />
                <directionalLight position={[10, 10, 5]} intensity={1} castShadow />
                <Environment preset="sweet" /> {/* 💡 かわいい雰囲気の環境光 */}

                {/* 3D棒グラフを描画 */}
                {dentalLogData.map((data, index) => (
                    <GraphBar key={index} data={data} index={index} />
                ))}

                {/* かわいいキャラクターを描画 */}
                <CuteCharacter />

                {/* マウスで視点を動かせる */}
                <OrbitControls />

                {/* 地面を追加 */}
                <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.1, 0]} receiveShadow>
                    <planeGeometry args={[50, 50]} />
                    <meshStandardMaterial color="#eeeeee" />
                </mesh>
            </Canvas>
        </div>
    );
};

export default CuteGraphScene;