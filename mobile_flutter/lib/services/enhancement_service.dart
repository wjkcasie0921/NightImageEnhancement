import 'dart:isolate';
import 'dart:typed_data';

import '../inference/ncnn_runner.dart';

/// 推理服务：通过异步边界避免在 UI 线程内堆积耗时任务。
///
/// 说明：真正的模型推理仍发生在 Android 原生 NCNN 层；这里使用
/// `Isolate.run` 将结果整理/错误包装与 UI 线程解耦，避免连续调用时
/// 让 Dart 事件循环被长时间同步逻辑阻塞。
class EnhancementService {
  final int inputSize;
  final NcnnRunner runner;

  EnhancementService({
    required this.inputSize,
    required this.runner,
  });

  Future<Uint8List> enhance(Uint8List bytes) async {
    try {
      final result = await runner.enhance(bytes);
      return await Isolate.run(() => Uint8List.fromList(result));
    } catch (e) {
      throw Exception('Enhancement failed: $e');
    }
  }
}
