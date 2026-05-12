import 'dart:typed_data';

import 'package:image/image.dart' as img;

/// 图片预处理
/// 目标：
/// 1. 使用 image 4.x API（pixel.r / pixel.g / pixel.b）
/// 2. 大图先降采样，防止内存爆炸
/// 3. 默认按 /255.0 归一化，后续如训练使用 mean/std 可替换
class ImagePreprocessor {
  final int inputSize;
  final int maxLongSide;

  ImagePreprocessor({
    required this.inputSize,
    this.maxLongSide = 2048,
  });

  Future<Uint8List> _downsampleIfNeeded(Uint8List bytes) async {
    final decoded = img.decodeImage(bytes);
    if (decoded == null) {
      throw Exception('Failed to decode image');
    }

    final longSide = decoded.width > decoded.height ? decoded.width : decoded.height;
    if (longSide <= maxLongSide) {
      return bytes;
    }

    // 先缩放再重新编码，降低高分辨率图片导致的 OOM 风险
    final ratio = maxLongSide / longSide;
    final targetW = (decoded.width * ratio).round();
    final targetH = (decoded.height * ratio).round();
    final resized = img.copyResize(
      decoded,
      width: targetW,
      height: targetH,
      interpolation: img.Interpolation.linear,
    );

    final compressed = img.encodeJpg(resized, quality: 95);
    return Uint8List.fromList(compressed);
  }

  /// 输出 NHWC Float32 tensor: [1, H, W, 3]
  Future<Float32List> preprocess(Uint8List bytes) async {
    final safeBytes = await _downsampleIfNeeded(bytes);
    final decoded = img.decodeImage(safeBytes);
    if (decoded == null) {
      throw Exception('Failed to decode image after downsampling');
    }

    final resized = img.copyResize(
      decoded,
      width: inputSize,
      height: inputSize,
      interpolation: img.Interpolation.linear,
    );

    final data = Float32List(1 * inputSize * inputSize * 3);
    var idx = 0;

    for (int y = 0; y < inputSize; y++) {
      for (int x = 0; x < inputSize; x++) {
        final pixel = resized.getPixel(x, y);

        // image 4.x API：pixel.r / pixel.g / pixel.b
        data[idx++] = pixel.r / 255.0;
        data[idx++] = pixel.g / 255.0;
        data[idx++] = pixel.b / 255.0;
      }
    }

    // 当前默认 /255.0 归一化
    // 如果你训练时使用了 mean/std，请在这里替换成对应标准化方式。
    return data;
  }
}
