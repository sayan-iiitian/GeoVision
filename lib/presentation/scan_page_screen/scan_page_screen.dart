import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:sayan_s_application11/core/app_export.dart';
import 'package:sayan_s_application11/presentation/micro_one_screen/micro_one_screen.dart';
import 'package:sayan_s_application11/presentation/micro_screen/micro_screen.dart';
import 'package:sayan_s_application11/routes/app_routes.dart';
import 'package:sayan_s_application11/widgets/custom_elevated_button.dart';
import 'package:sayan_s_application11/widgets/custom_icon_button.dart';
import 'package:image_picker/image_picker.dart';

class ScanPageScreen extends StatelessWidget {
  final ImagePicker _imagePicker = ImagePicker();

  ScanPageScreen({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    mediaQueryData = MediaQuery.of(context);
    return SafeArea(
      child: Scaffold(
        extendBody: true,
        extendBodyBehindAppBar: true,
        backgroundColor: appTheme.black900.withOpacity(0.15),
        body: Container(
          width: mediaQueryData.size.width,
          height: mediaQueryData.size.height,
          decoration: BoxDecoration(
            color: appTheme.black900.withOpacity(0.15),
            image: DecorationImage(
              image: AssetImage(ImageConstant.imgScanPage),
              fit: BoxFit.cover,
            ),
          ),
          child: Container(
            width: double.maxFinite,
            padding: EdgeInsets.symmetric(horizontal: 23.h, vertical: 32.v),
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
                Spacer(flex: 49),
                GestureDetector(
                  onTap: () {
                    _pickImageFromGallery(context);
                  },
                  child: Container(
                    margin: EdgeInsets.only(left: 114.h, right: 89.h),
                    padding: EdgeInsets.all(10.h),
                    decoration: AppDecoration.outlineBlack,
                    child: Container(
                      padding:
                          EdgeInsets.symmetric(horizontal: 19.h, vertical: 6.v),
                      decoration: AppDecoration.outlineBlack900.copyWith(
                        borderRadius: BorderRadiusStyle.roundedBorder8,
                      ),
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          SizedBox(height: 2.v),
                          SizedBox(
                            width: 120.h,
                            child: Text(
                              "Upload Microscopic\n Image",
                              maxLines: 2,
                              overflow: TextOverflow.ellipsis,
                              textAlign: TextAlign.center,
                              style: theme.textTheme.labelLarge,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
                SizedBox(height: 35.v),
                CustomElevatedButton(
                  width: 159.h,
                  text: "Upload Macro Image",
                  margin: EdgeInsets.only(right: 99.h),
                  buttonTextStyle: theme.textTheme.labelLarge!,
                  onPressed: () {
                    onTapUploadMacroImage(context);
                  },
                ),
                Spacer(flex: 50),
              ],
            ),
          ),
        ),
      ),
    );
  }

  onTapBtnArrowLeft(BuildContext context) {
    Navigator.pop(context);
  }

  onTapUploadMacroImage(BuildContext context) async {
    try {
      final XFile? pickedImage = await _imagePicker.pickImage(
        source: ImageSource.gallery,
        imageQuality: 80,
      );

      if (pickedImage != null) {
        // Pass the selected image to MicroScreen
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) =>
                MicroOneScreen(imageFile: File(pickedImage.path)),
          ),
        );
      }
    } catch (e) {
      print("Error picking image: $e");
    }
  }

  Future<void> _pickImageFromGallery(BuildContext context) async {
    try {
      final XFile? pickedImage = await _imagePicker.pickImage(
        source: ImageSource.gallery,
        imageQuality: 80,
      );

      if (pickedImage != null) {
        // Pass the selected image to MicroScreen
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) =>
                MicroScreen(imageFile: File(pickedImage.path)),
          ),
        );
      }
    } catch (e) {
      print("Error picking image: $e");
    }
  }
}
