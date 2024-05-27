import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:flutter/material.dart';
import 'package:sayan_s_application11/core/app_export.dart';
import 'package:sayan_s_application11/widgets/custom_elevated_button.dart';
import 'package:sayan_s_application11/widgets/custom_icon_button.dart';

class MicroScreen extends StatefulWidget {
  final File imageFile;

  const MicroScreen({Key? key, required this.imageFile}) : super(key: key);

  @override
  State<MicroScreen> createState() => _MicroScreenState();
}

class _MicroScreenState extends State<MicroScreen> {
  Image? roundnessImage;

  Future<void> uploadGrain(File imageFile) async {
    var apiUrl = Uri.parse('http://192.168.84.9:5000/grain_char');
    var request = http.MultipartRequest('POST', apiUrl);
    var image = await http.MultipartFile.fromPath('image', imageFile.path);
    request.files.add(image);

    try {
      var response = await request.send();

      if (response.statusCode == 200) {
        print('Image uploaded successfully!');
      } else {
        print('Failed to upload image. Status code: ${response.statusCode}');
      }
    } catch (error) {
      print('Error uploading image: $error');
    }
  }

  Future<void> fetchRoundnessHistogram(BuildContext context) async {
    var apiUrl = Uri.parse('http://192.168.84.9:5000/roundness');

    try {
      var response = await http.get(apiUrl);

      if (response.statusCode == 200) {
        var imagePath = json.decode(response.body)['image_path'];

        // Show the roundness histogram image in a popup dialog
        showDialog(
          context: context,
          builder: (BuildContext context) {
            return AlertDialog(
              title: Text('Roundness Histogram'),
              content: Container(
                child: Image.network(imagePath),
              ),
              actions: <Widget>[
                TextButton(
                  onPressed: () {
                    Navigator.of(context).pop();
                  },
                  child: Text('Close'),
                ),
              ],
            );
          },
        );

        print('Received roundness histogram image!');
      } else {
        print(
          'Failed to fetch roundness histogram image. Status code: ${response.statusCode}',
        );
      }
    } catch (error) {
      print('Error fetching roundness histogram image: $error');
    }
  }

  Future<void> fetchCompactnessResults(BuildContext context) async {
    final apiUrl = Uri.parse('http://192.168.84.9:5000/compactness');

    try {
      var response = await http.get(apiUrl);

      if (response.statusCode == 200) {
        final Map<String, dynamic> compactnessResults =
            json.decode(response.body);

        // Show compactness results in a pop-up dialog
        showDialog(
          context: context,
          builder: (BuildContext context) {
            return AlertDialog(
              title: Text('Compactness Results'),
              content: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: compactnessResults.entries.map((entry) {
                  return Text('${entry.key}: ${entry.value}');
                }).toList(),
              ),
              actions: <Widget>[
                TextButton(
                  onPressed: () {
                    Navigator.of(context).pop();
                  },
                  child: Text('Close'),
                ),
              ],
            );
          },
        );
      } else {
        // Handle errors
        showDialog(
          context: context,
          builder: (BuildContext context) {
            return AlertDialog(
              title: Text('Error'),
              content: Text('Failed to fetch compactness results.'),
              actions: <Widget>[
                TextButton(
                  onPressed: () {
                    Navigator.of(context).pop();
                  },
                  child: Text('Close'),
                ),
              ],
            );
          },
        );
      }
    } catch (error) {
      print('Error fetching compactness results: $error');
    }
  }

  @override
  Widget build(BuildContext context) {
    mediaQueryData = MediaQuery.of(context);
    return SafeArea(
      child: Scaffold(
        body: SizedBox(
          width: double.maxFinite,
          child: Container(
            width: double.maxFinite,
            padding: EdgeInsets.symmetric(horizontal: 23.h, vertical: 32.v),
            decoration: AppDecoration.fillBlack.copyWith(
              image: DecorationImage(
                image: AssetImage(ImageConstant.imgScanPage),
                fit: BoxFit.cover,
              ),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                CustomIconButton(
                  height: 60.adaptSize,
                  width: 60.adaptSize,
                  padding: EdgeInsets.all(14.h),
                  alignment: Alignment.centerLeft,
                  onTap: () {
                    onTapBtnArrowLeft(context);
                  },
                  child: CustomImageView(imagePath: ImageConstant.imgArrowLeft),
                ),
                Spacer(),
                _buildPreviewImage(context),
                SizedBox(height: 29.v),
                _buildGrainDetails(context),
                SizedBox(height: 24.v),
                _buildRoundness(context),
                SizedBox(height: 37.v),
                _buildCompactness(context),
                SizedBox(
                    height: 24.v), // Adjusted spacing for the Sorting button
                _buildSorting(context),
                SizedBox(
                    height: 65.v), // Adjusted spacing after the Sorting button
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildPreviewImage(BuildContext context) {
    return Container(
      width: mediaQueryData.size.width - 48.h,
      height: mediaQueryData.size.height * 0.4,
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(12.h),
        image: DecorationImage(
          image: FileImage(widget.imageFile),
          fit: BoxFit.cover,
        ),
      ),
    );
  }

  Widget _buildGrainDetails(BuildContext context) {
    return CustomElevatedButton(
      width: 159.h,
      text: "Grain Details",
      margin: EdgeInsets.only(right: 96.h),
      onPressed: () {
        uploadGrain(widget.imageFile);
      },
    );
  }

  Widget _buildRoundness(BuildContext context) {
    return CustomElevatedButton(
      width: 159.h,
      text: "Roundness",
      margin: EdgeInsets.only(right: 96.h),
      onPressed: () {
        fetchRoundnessHistogram(context);
      },
    );
  }

  Widget _buildCompactness(BuildContext context) {
    return CustomElevatedButton(
      width: 159.h,
      text: "Compactness",
      margin: EdgeInsets.only(right: 96.h),
      onPressed: () {
        fetchCompactnessResults(context);
      },
    );
  }

  Widget _buildSorting(BuildContext context) {
    return CustomElevatedButton(
      width: 159.h,
      text: "Sorting",
      margin: EdgeInsets.only(right: 96.h),
      onPressed: () {
        fetchSortingResults(context);
      },
    );
  }

  Future<void> fetchSortingResults(BuildContext context) async {
    var apiUrl = Uri.parse('http://192.168.84.9:5000/sorting');

    try {
      var response = await http.get(apiUrl);

      if (response.statusCode == 200) {
        final Map<String, dynamic> sortingResults = json.decode(response.body);

        // Show sorting results in a pop-up dialog
        showDialog(
          context: context,
          builder: (BuildContext context) {
            return AlertDialog(
              title: Text('Sorting Results'),
              content: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: sortingResults.entries.map((entry) {
                  return Text('${entry.key}: ${entry.value}');
                }).toList(),
              ),
              actions: <Widget>[
                TextButton(
                  onPressed: () {
                    Navigator.of(context).pop();
                  },
                  child: Text('Close'),
                ),
              ],
            );
          },
        );
      } else {
        // Handle errors
        showDialog(
          context: context,
          builder: (BuildContext context) {
            return AlertDialog(
              title: Text('Error'),
              content: Text('Failed to fetch sorting results.'),
              actions: <Widget>[
                TextButton(
                  onPressed: () {
                    Navigator.of(context).pop();
                  },
                  child: Text('Close'),
                ),
              ],
            );
          },
        );
      }
    } catch (error) {
      print('Error fetching sorting results: $error');
    }
  }

  onTapBtnArrowLeft(BuildContext context) {
    Navigator.pop(context);
  }
}
