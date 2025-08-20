import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
import io
import gc  # 가비지 컬렉션 추가

# 데이터 로드 함수
def load_data(json_path, bin_path, visualize_once=False):
    """
    JSON과 BIN 파일에서 데이터를 로드합니다.
    visualize_once=True일 경우 첫 번째 이미지를 시각화하고 프로그램을 종료합니다.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 센서 데이터 추출
    try:
        sensor_data_json = data['sensor_data'][0]
        sensor_values = [
            float(sensor_data_json['NTC'][0]['value']),
            float(sensor_data_json['PM10'][0]['value']),
            float(sensor_data_json['PM2.5'][0]['value']),
            float(sensor_data_json['PM1.0'][0]['value']),
            float(sensor_data_json['CT1'][0]['value']),
            float(sensor_data_json['CT2'][0]['value']),
            float(sensor_data_json['CT3'][0]['value']),
            float(sensor_data_json['CT4'][0]['value']),
        ]
    except (KeyError, IndexError) as e:
        print(f"센서 데이터 읽기 오류 in {json_path}: {e}")
        return None, None, None, None

    # 열화상 데이터 로드
    try:
        ir_img = np.load(bin_path)

        if visualize_once:
            img_normalized = cv2.normalize(ir_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            img_colored = cv2.applyColorMap(img_normalized, cv2.COLORMAP_JET)
            cv2.imshow(f'Thermal Image - {os.path.basename(bin_path)}', img_colored)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print("시각화 완료 후 프로그램을 종료합니다.")
            exit()
    except FileNotFoundError:
        print(f"BIN 파일 없음: {bin_path}")
        return None, None, None, None
    except Exception as e:
        print(f"BIN 파일 로드 오류 in {bin_path}: {e}")
        return None, None, None, None

    # 레이블(상태) 추출
    try:
        label = int(data['annotations'][0]['tagging'][0]['state'])
    except (KeyError, IndexError) as e:
        print(f"레이블 데이터 읽기 오류 in {json_path}: {e}")
        return None, None, None, None

    # device_id 추가
    try:
        device_id = data['meta_info'][0]['device_id']
    except (KeyError, IndexError) as e:
        print(f"device_id 읽기 오류 in {json_path}: {e}")
        return None, None, None, None

    return sensor_values, ir_img, label, device_id

# 슬라이딩 윈도우 생성 함수
def create_sliding_windows(data_list, window_size=30, step_size=1):
    if not data_list or len(data_list) < window_size:
        return [], [], []
    sensor_windows, ir_windows, label_windows = [], [], []
    for i in range(0, len(data_list) - window_size + 1, step_size):
        sensor_window = np.array([entry[0] for entry in data_list[i:i + window_size]])
        ir_window = np.array([entry[1] for entry in data_list[i:i + window_size]])
        label = data_list[i + window_size - 1][2]
        sensor_windows.append(sensor_window)
        ir_windows.append(ir_window)
        label_windows.append(label)
    return np.array(sensor_windows), np.array(ir_windows), np.array(label_windows)

# 개별 디렉토리에서 데이터 처리
def process_directory(source_base_dir, label_sub_dir_path, window_size=30, step_size=1, visualize_first=False):
    data_list = []

    try:
        label_files = sorted(os.listdir(label_sub_dir_path))
    except FileNotFoundError:
        print(f"라벨링 폴더 없음: {label_sub_dir_path}")
        return []

    is_first_file = True

    for label_file in label_files:
        if not label_file.endswith('.json'):
            continue

        json_path = os.path.join(label_sub_dir_path, label_file)
        sub_dir_name = os.path.basename(label_sub_dir_path)
        source_sub_dir_name = sub_dir_name.replace('TL_', 'TS_')
        bin_file_name = label_file.replace('TL_', 'TS_').replace('.json', '.bin')
        bin_path = os.path.join(source_base_dir, source_sub_dir_name, bin_file_name)

        should_visualize = visualize_first and is_first_file
        sensor_data, ir_img, label, device_id = load_data(json_path, bin_path, visualize_once=should_visualize)

        try:
            timestamp_str = '_'.join(label_file.split('_')[-2:]).replace('.json', '')
        except:
            timestamp_str = "0000_000000"

        if sensor_data is not None and ir_img is not None and label is not None:
            data_list.append({'sensor': sensor_data, 'ir': ir_img, 'label': label, 'device_id': device_id, 'timestamp': timestamp_str})
            is_first_file = False

    return data_list

# 요약 이미지 생성 함수
def create_summary_image(sensor_sample, ir_sample, output_path):
    """학습 데이터셋의 첫 샘플로 요약 이미지를 생성합니다."""

    thermal_image = cv2.normalize(ir_sample, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    thermal_image_color = cv2.applyColorMap(thermal_image, cv2.COLORMAP_JET)
    thermal_image_resized = cv2.resize(thermal_image_color, (480, 480))

    sensor_labels = ['NTC', 'PM10', 'PM2.5', 'PM1.0', 'CT1', 'CT2', 'CT3', 'CT4']

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.bar(sensor_labels, sensor_sample, color='skyblue')
    ax.set_title('Sample Sensor Data', fontsize=16)
    ax.set_ylabel('Values')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    graph_img = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), 1)
    graph_img_resized = cv2.resize(graph_img, (480, 480))
    plt.close(fig)

    canvas = np.ones((540, 980, 3), dtype=np.uint8) * 255
    canvas[50:530, 10:490] = thermal_image_resized
    canvas[50:530, 500:980] = graph_img_resized

    cv2.putText(canvas, 'Thermal Image Sample', (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(canvas, 'Sensor Data Sample (Bar Chart)', (500, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

    cv2.imwrite(output_path, canvas)
    print(f"\n요약 이미지가 다음 경로에 저장되었습니다: {output_path}")

# 메모리 효율적인 데이터 처리 및 저장
def process_and_save_all_data(source_base_dir, label_base_dir, output_dir, window_size=30, step_size=1, visualize_first_image=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 임시 저장 디렉토리 생성
    temp_dir = os.path.join(output_dir, 'temp_device_data')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    label_sub_dirs = sorted([d for d in os.listdir(label_base_dir) if os.path.isdir(os.path.join(label_base_dir, d))])

    device_data_map = {}  # 장비별 데이터 경로 저장
    total_windows_per_device = {}  # 장비별 윈도우 개수 추적

    # 1단계: 폴더별로 데이터 로드하고 장비별로 임시 저장
    for sub_dir_idx, sub_dir_name in enumerate(label_sub_dirs):
        print(f"--- [{sub_dir_idx+1}/{len(label_sub_dirs)}] {sub_dir_name} 폴더 데이터 로드 중... ---")
        label_sub_dir_path = os.path.join(label_base_dir, sub_dir_name)
        should_visualize = visualize_first_image and sub_dir_idx == 0
        folder_data = process_directory(source_base_dir, label_sub_dir_path, visualize_first=should_visualize)

        # 장비별로 그룹화
        for item in folder_data:
            device_id = item['device_id']
            if device_id not in device_data_map:
                device_data_map[device_id] = []
            device_data_map[device_id].append(item)

        # 메모리 정리
        del folder_data
        gc.collect()

    # 2단계: 장비별로 처리하고 임시 파일로 저장
    print("\n--- 장비별 슬라이딩 윈도우 생성 및 임시 저장 ---")
    for device_idx, (device_id, device_data) in enumerate(device_data_map.items()):
        print(f"[{device_idx+1}/{len(device_data_map)}] {device_id} 장비 처리 중...")

        # 시간순 정렬
        device_data.sort(key=lambda x: x['timestamp'])

        # 슬라이딩 윈도우 생성
        formatted_data = [(d['sensor'], d['ir'], d['label']) for d in device_data]
        sensor_windows, ir_windows, labels = create_sliding_windows(formatted_data, window_size, step_size)

        if len(labels) > 0:
            # 장비별 임시 파일 저장
            device_file_prefix = os.path.join(temp_dir, f"device_{device_id}")
            np.save(f"{device_file_prefix}_sensor.npy", sensor_windows)
            np.save(f"{device_file_prefix}_ir.npy", ir_windows)
            np.save(f"{device_file_prefix}_labels.npy", labels)

            total_windows_per_device[device_id] = len(labels)
            print(f"  └─ {device_id}: {len(labels)}개 윈도우 저장 완료")

        # 메모리 정리
        del device_data, formatted_data, sensor_windows, ir_windows, labels
        gc.collect()

    # 3단계: 임시 파일들을 읽어서 train/val/test로 분할
    print("\n--- 전체 데이터 통합 및 분할 ---")

    # 전체 데이터 크기 계산
    total_windows = sum(total_windows_per_device.values())
    print(f"전체 윈도우 개수: {total_windows}")

    # 모든 장비 데이터 로드 및 통합
    all_sensor_data = []
    all_ir_data = []
    all_labels = []

    for device_id in total_windows_per_device.keys():
        device_file_prefix = os.path.join(temp_dir, f"device_{device_id}")
        sensor_data = np.load(f"{device_file_prefix}_sensor.npy")
        ir_data = np.load(f"{device_file_prefix}_ir.npy")
        labels = np.load(f"{device_file_prefix}_labels.npy")

        all_sensor_data.append(sensor_data)
        all_ir_data.append(ir_data)
        all_labels.append(labels)

        # 메모리 정리
        del sensor_data, ir_data, labels
        gc.collect()

    # 데이터 통합
    final_sensor_data = np.vstack(all_sensor_data)
    final_ir_data = np.vstack(all_ir_data)
    final_labels = np.concatenate(all_labels)

    # 메모리 정리
    del all_sensor_data, all_ir_data, all_labels
    gc.collect()

    # 데이터 분할
    train_sensor, test_sensor, train_ir, test_ir, train_labels, test_labels = train_test_split(
        final_sensor_data, final_ir_data, final_labels, test_size=0.2, random_state=42, stratify=final_labels
    )
    val_sensor, test_sensor, val_ir, test_ir, val_labels, test_labels = train_test_split(
        test_sensor, test_ir, test_labels, test_size=0.5, random_state=42, stratify=test_labels
    )

    # 4단계: 최종 데이터 저장
    datasets = {
        'train': (train_sensor, train_ir, train_labels),
        'val': (val_sensor, val_ir, val_labels),
        'test': (test_sensor, test_ir, test_labels)
    }

    print("\n--- 💾 최종 저장 ---")
    for name, (sensor, ir, labels) in datasets.items():
        sensor_path = os.path.join(output_dir, f"{name}_sensor.npy")
        ir_path = os.path.join(output_dir, f"{name}_ir.npy")
        labels_path = os.path.join(output_dir, f"{name}_labels.npy")

        np.save(sensor_path, sensor)
        np.save(ir_path, ir)
        np.save(labels_path, labels)

        print(f"[{name.upper()}] 저장 완료 - 샘플 수: {len(labels)}")

    # 요약 이미지 생성
    summary_sensor_sample = datasets['train'][0][0][-1]
    summary_ir_sample = datasets['train'][1][0][-1]
    create_summary_image(summary_sensor_sample, summary_ir_sample, os.path.join(output_dir, 'summary_visualization.png'))

    # 5단계: 임시 파일 정리
    print("\n--- 임시 파일 정리 중... ---")
    for device_id in total_windows_per_device.keys():
        device_file_prefix = os.path.join(temp_dir, f"device_{device_id}")
        for suffix in ['_sensor.npy', '_ir.npy', '_labels.npy']:
            file_path = f"{device_file_prefix}{suffix}"
            if os.path.exists(file_path):
                os.remove(file_path)
    os.rmdir(temp_dir)

    print("\n✅ 모든 처리가 완료되었습니다!")

if __name__ == "__main__":
    source_base_dir = '/home/juyoung-lab/ws/dev_ws/pi3/datasets/MPTMS/data_MPTMS/67.제조현장 이송장치의 열화 예지보전 멀티모달 데이터/3.개방데이터/1.데이터/Training/01.원천데이터'
    label_base_dir = '/home/juyoung-lab/ws/dev_ws/pi3/datasets/MPTMS/data_MPTMS/67.제조현장 이송장치의 열화 예지보전 멀티모달 데이터/3.개방데이터/1.데이터/Training/02.라벨링데이터'
    output_dir = '/media/juyoung-lab/HDD-4TB1/target_pi3/preprocessed_data'
    VISUALIZE_AND_DEBUG = False

    process_and_save_all_data(
        source_base_dir=source_base_dir,
        label_base_dir=label_base_dir,
        output_dir=output_dir,
        visualize_first_image=VISUALIZE_AND_DEBUG
    )
