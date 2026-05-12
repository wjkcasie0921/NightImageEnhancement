import 'dart:typed_data';

import 'package:saver_gallery/saver_gallery.dart';

class SaveService {
  static Future<void> saveToGallery(Uint8List bytes) async {
    await SaverGallery.saveImage(
      bytes,
      quality: 95,
      fileName: 'retinex_enhanced_${DateTime.now().millisecondsSinceEpoch}.jpg',
      androidRelativePath: 'Pictures/RetinexNetLite',
      skipIfExists: false,
    );
  }
}
