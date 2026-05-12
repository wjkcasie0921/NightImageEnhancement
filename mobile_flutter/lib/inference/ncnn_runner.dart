import 'dart:typed_data';

import 'package:flutter/services.dart';

/// NCNN 推理封装
/// 使用 MethodChannel 调用 Android 原生 Kotlin 实现
class NcnnRunner {
  static const MethodChannel _channel = MethodChannel('retinex_ncnn');

  Future<void> loadModel({
    required String paramPath,
    required String binPath,
  }) async {
    await _channel.invokeMethod('loadModel', {
      'paramPath': paramPath,
      'binPath': binPath,
    });
  }

  /// inputBytes: 原始图片字节
  /// 返回：增强后的 JPEG/PNG 字节流
  Future<Uint8List> enhance(Uint8List inputBytes) async {
    final result = await _channel.invokeMethod<Uint8List>(
      'enhance',
      {'imageBytes': inputBytes},
    );
    if (result == null) {
      throw StateError('NCNN enhance returned null');
    }
    return result;
  }

  Future<void> close() async {
    await _channel.invokeMethod('close');
  }
}
