import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';

import '../services/image_picker_service.dart';
import '../services/save_service.dart';
import '../widgets/before_after_slider.dart';
import '../inference/ncnn_runner.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  static const String _paramAsset = 'assets/models/retinex_lite.ncnn.param';
  static const String _binAsset = 'assets/models/retinex_lite.ncnn.bin';
  String? _paramPath;
  String? _binPath;

  final ImagePickerService _pickerService = ImagePickerService();
  final NcnnRunner _runner = NcnnRunner();

  Uint8List? _before;
  Uint8List? _after;
  bool _loading = false;
  bool _initialized = false;
  bool _initializing = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _initModel();
    });
  }

  Future<String> _copyAssetToLocalFile(String assetPath) async {
    final bytes = await rootBundle.load(assetPath);
    final supportDir = await getApplicationSupportDirectory();
    final file = File('${supportDir.path}/${assetPath.split('/').last}');
    if (!await file.exists() || await file.length() != bytes.lengthInBytes) {
      await file.writeAsBytes(bytes.buffer.asUint8List(), flush: true);
    }
    return file.path;
  }

  Future<void> _initModel() async {
    if (_initialized || _initializing) return;
    _initializing = true;
    try {
      final messenger = ScaffoldMessenger.maybeOf(context);
      messenger?.showSnackBar(
        const SnackBar(content: Text('正在加载模型，请稍候...')),
      );

      _paramPath ??= await _copyAssetToLocalFile(_paramAsset);
      _binPath ??= await _copyAssetToLocalFile(_binAsset);

      await _runner.loadModel(
        paramPath: _paramPath!,
        binPath: _binPath!,
      );
      if (!mounted) return;
      setState(() => _initialized = true);
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('模型加载完成')),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('模型加载失败：$e')),
      );
    } finally {
      _initializing = false;
    }
  }

  Future<void> _requestGalleryPermission() async {
    final photos = await Permission.photos.request();
    if (photos.isGranted) return;
    final storage = await Permission.storage.request();
    if (storage.isGranted) return;
    throw Exception('相册权限未授予');
  }

  Future<void> _requestCameraPermission() async {
    final status = await Permission.camera.request();
    if (!status.isGranted) throw Exception('相机权限未授予');
  }

  Future<void> _pickFromGallery() async {
    try {
      await _requestGalleryPermission();
      final bytes = await _pickerService.pickFromGallery();
      if (bytes == null) return;
      setState(() {
        _before = bytes;
        _after = null;
      });
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('选择图片失败：$e')),
      );
    }
  }

  Future<void> _pickFromCamera() async {
    try {
      await _requestCameraPermission();
      final bytes = await _pickerService.pickFromCamera();
      if (bytes == null) return;
      setState(() {
        _before = bytes;
        _after = null;
      });
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('拍照失败：$e')),
      );
    }
  }

  Future<void> _enhance() async {
    if (_before == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('请先选择一张图片')),
      );
      return;
    }
    if (!_initialized) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('模型尚未加载完成，请稍后重试')),
      );
      return;
    }

    setState(() => _loading = true);
    try {
      final result = await _runner.enhance(_before!);
      setState(() => _after = result);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('增强失败：$e')),
      );
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  Future<void> _save() async {
    if (_after == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('没有可保存的增强结果')),
      );
      return;
    }

    try {
      await SaveService.saveToGallery(_after!);
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('保存成功')),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('保存失败：$e')),
      );
    }
  }

  @override
  void dispose() {
    _runner.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final readyText = _initialized ? '模型已就绪' : '模型加载中...';

    return Scaffold(
      appBar: AppBar(
        title: const Text('RetinexNetLite 增强'),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 12),
            child: Center(child: Text(readyText)),
          ),
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: BeforeAfterSlider(before: _before, after: _after),
          ),
          if (_loading) const LinearProgressIndicator(),
          Padding(
            padding: const EdgeInsets.all(16),
            child: Wrap(
              spacing: 12,
              runSpacing: 12,
              children: [
                ElevatedButton(onPressed: _pickFromGallery, child: const Text('相册选择')),
                ElevatedButton(onPressed: _pickFromCamera, child: const Text('拍照')),
                ElevatedButton(onPressed: _enhance, child: const Text('一键增强')),
                ElevatedButton(onPressed: _save, child: const Text('保存结果')),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
