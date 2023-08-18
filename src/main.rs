#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use eframe::egui;
use std::mem;

fn main() -> Result<(), eframe::Error> {
    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).
    let options = eframe::NativeOptions {
        initial_window_size: Some(egui::vec2(900.0, 300.0)),
        ..Default::default()
    };
    eframe::run_native(
        "Bits Tool",
        options,
        Box::new(|_cc| Box::<MyApp>::default()),
    )
}

struct MyApp {
    a_bools: [bool; 32],
    a_u32: u32,
    a_i32: i32,
    a_f32: f32,
    a_hex_str: String,
    a_u32_str: String,
    a_i32_str: String,
    a_f32_str: String,
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            a_bools: [false; 32],
            a_u32: 0,
            a_i32: 0,
            a_f32: 0.0,
            a_hex_str: "0x0".to_owned(),
            a_u32_str: 0.to_string(),
            a_i32_str: 0.to_string(),
            a_f32_str: 0.0.to_string(),
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                // signed bit
                ui.group(|ui| {
                    ui.vertical(|ui| {
                        ui.label("sign");
                        for i in 0..1 {
                            ui.vertical(|ui| {
                                if ui.checkbox(&mut self.a_bools[31 - i], "").clicked() {
                                    if self.a_bools[31 - i] {
                                        self.a_u32 = self.a_u32 | (0x1 << (31 - i));
                                    } else {
                                        self.a_u32 = self.a_u32 & !(0x1 << (31 - i));
                                    }
                                }
                                ui.label(format!("{}", 31 - i));
                            });
                        }
                    })
                });

                // exp bits
                ui.group(|ui| {
                    ui.vertical(|ui| {
                        ui.label("exp");
                        ui.horizontal(|ui| {
                            for i in 1..9 {
                                ui.vertical(|ui| {
                                    if ui.checkbox(&mut self.a_bools[31 - i], "").clicked() {
                                        if self.a_bools[31 - i] {
                                            self.a_u32 = self.a_u32 | (0x1 << (31 - i));
                                        } else {
                                            self.a_u32 = self.a_u32 & !(0x1 << (31 - i));
                                        }
                                    }
                                    ui.label(format!("{}", 31 - i));
                                });
                            }
                        });
                    });
                });

                // exp mantissa
                ui.group(|ui| {
                    ui.vertical(|ui| {
                        ui.label("mantissa");
                        ui.horizontal(|ui| {
                            for i in 9..32 {
                                ui.vertical(|ui| {
                                    if ui.checkbox(&mut self.a_bools[31 - i], "").clicked() {
                                        if self.a_bools[31 - i] {
                                            self.a_u32 = self.a_u32 | (0x1 << (31 - i));
                                        } else {
                                            self.a_u32 = self.a_u32 & !(0x1 << (31 - i));
                                        }
                                    }
                                    ui.label(format!("{}", 31 - i));
                                });
                            }
                        });
                    });
                });
            });

            if ui.text_edit_singleline(&mut self.a_hex_str).changed() {}
            if ui.text_edit_singleline(&mut self.a_u32_str).changed() {}
            if ui.text_edit_singleline(&mut self.a_i32_str).changed() {}
            if ui.text_edit_singleline(&mut self.a_f32_str).changed() {}

            ui.horizontal(|ui| {
                if ui.button("-1").clicked() {
                    unsafe {
                        self.a_u32 = mem::transmute(-1);
                    }
                }
                if ui.button("0").clicked() {
                    self.a_u32 = 0;
                }

                if ui.button("des").clicked() {
                    if self.a_u32 == 0 {
                        self.a_u32 = u32::MAX;
                    } else {
                        self.a_u32 = self.a_u32 - 1;
                    }
                }
                if ui.button("inc").clicked() {
                    if self.a_u32 == u32::MAX {
                        self.a_u32 = 0;
                    } else {
                        self.a_u32 = self.a_u32 + 1;
                    }
                }

                if ui.button("lshr").clicked() {
                    self.a_u32 = self.a_u32.checked_shr(1).unwrap();
                }

                if ui.button("lshl").clicked() {
                    self.a_u32 = self.a_u32.checked_shl(1).unwrap();
                }

                if ui.button("ashr").clicked() {
                    unsafe {
                        let mut tmp: i32 = mem::transmute(self.a_u32);
                        tmp = tmp.checked_shr(1).unwrap();
                        self.a_u32 = mem::transmute(tmp);
                    }
                }
            });

            unsafe {
                self.a_i32 = mem::transmute(self.a_u32);
                self.a_f32 = mem::transmute(self.a_u32);
            }
            self.a_hex_str = format!("0x{:X}", self.a_u32);
            self.a_i32_str = format!("{}", self.a_i32);
            self.a_u32_str = format!("{}", self.a_u32);
            self.a_f32_str = format!("{}", self.a_f32);
            for i in 0..32 {
                if self.a_u32 & (0x1 << i) != 0 {
                    self.a_bools[i] = true;
                } else {
                    self.a_bools[i] = false;
                }
            }
        });
    }
}
