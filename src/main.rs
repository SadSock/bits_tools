#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use eframe::egui;
use std::{mem, sync::atomic::AtomicU32};

fn main() -> Result<(), eframe::Error> {
    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).
    let options = eframe::NativeOptions {
        initial_window_size: Some(egui::vec2(850.0, 240.0)),
        ..Default::default()
    };
    eframe::run_native(
        "My egui App",
        options,
        Box::new(|_cc| Box::<MyApp>::default()),
    )
}

struct MyApp {
    d32_array: [bool; 32],
    num_u32: u32,
    num_i32: i32,
    num_f32: f32,
    hex: String,
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            d32_array: [false; 32],
            num_u32: 0,
            num_i32: 0,
            num_f32: 0.0,
            hex: "0x0".to_owned(),
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("My egui Application");

            ui.horizontal(|ui| {
                ui.group(|ui| {
                    // signed bit
                    ui.vertical(|ui| {
                        ui.label("sign");
                        for i in 0..1 {
                            ui.vertical(|ui| {
                                if ui.checkbox(&mut self.d32_array[31 - i], "").clicked() {
                                    if self.d32_array[31 - i] {
                                        self.num_u32 = self.num_u32 | (0x1 << (31 - i));
                                    } else {
                                        self.num_u32 = self.num_u32 & !(0x1 << (31 - i));
                                    }
                                }
                                ui.label(format!("{}", 31 - i));
                            });
                        }
                    })
                });

                ui.group(|ui| {
                    ui.vertical(|ui| {
                        ui.label("exp");
                        // exp bits
                        ui.horizontal(|ui| {
                            for i in 1..9 {
                                ui.vertical(|ui| {
                                    if ui.checkbox(&mut self.d32_array[31 - i], "").clicked() {
                                        if self.d32_array[31 - i] {
                                            self.num_u32 = self.num_u32 | (0x1 << (31 - i));
                                        } else {
                                            self.num_u32 = self.num_u32 & !(0x1 << (31 - i));
                                        }
                                    }
                                    ui.label(format!("{}", 31 - i));
                                });
                            }
                        });
                    });
                });

                ui.group(|ui| {
                    ui.vertical(|ui| {
                        ui.label("mantiss");
                        // exp bits
                        ui.horizontal(|ui| {
                            for i in 9..32 {
                                ui.vertical(|ui| {
                                    if ui.checkbox(&mut self.d32_array[31 - i], "").clicked() {
                                        if self.d32_array[31 - i] {
                                            self.num_u32 = self.num_u32 | (0x1 << (31 - i));
                                        } else {
                                            self.num_u32 = self.num_u32 & !(0x1 << (31 - i));
                                        }
                                    }
                                    ui.label(format!("{}", 31 - i));
                                });
                            }
                        });
                    });
                });
            });
            if ui.text_edit_singleline(&mut self.hex).changed() {}

            if ui.button("-1").clicked() {
                self.num_i32 = -1;

                unsafe {
                    self.num_u32 = mem::transmute(self.num_i32);
                }

                for i in 0..32 {
                    self.d32_array[i] = true;
                }
            }
            if ui.button("0").clicked() {
                self.num_u32 = 0;
                self.num_i32 = 0;
                for i in 0..32 {
                    self.d32_array[i] = false;
                }
            }

            if ui.button("des").clicked() {
                if self.num_u32 == 0 {
                    self.num_u32 = u32::MAX;
                } else {
                    self.num_u32 = self.num_u32 - 1;
                }
            }

            if ui.button("inc").clicked() {
                if self.num_u32 == u32::MAX {
                    self.num_u32 = 0;
                } else {
                    self.num_u32 = self.num_u32 + 1;
                }
            }

            if ui.button("lshr").clicked() {
                self.num_u32 = self.num_u32.checked_shr(1).unwrap();
            }

            if ui.button("lshl").clicked() {
                self.num_u32 = self.num_u32.checked_shl(1).unwrap();
            }

            if ui.button("ashr").clicked() {
                unsafe {
                    let mut tmp: i32 = mem::transmute(self.num_u32);
                    tmp = tmp.checked_shr(1).unwrap();
                    self.num_u32 = mem::transmute(tmp);
                }
            }

            self.hex = format!("0x{:X}", self.num_u32);
            unsafe {
                // self.num_u32 = self.num_u32.unchecked_sub(1);
                self.num_i32 = mem::transmute(self.num_u32);
                self.num_f32 = mem::transmute(self.num_u32);
            }
            for i in 0..32 {
                if self.num_u32 & (0x1 << i) != 0 {
                    self.d32_array[i] = true;
                } else {
                    self.d32_array[i] = false;
                }
            }

            ui.label(format!("i32: {}", self.num_i32 % 8));
            ui.label(format!("hex: 0x{:X}", self.num_u32));
            ui.label(format!("u32: {}", self.num_u32));
            ui.label(format!("i32: {}", self.num_i32));
            ui.label(format!("f32: {}", self.num_f32));
        });
    }
}
