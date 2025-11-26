use rustuna_core::sampler::RandomSampler;
use rustuna_core::storage::InMemoryStorage;
use rustuna_core::study::{create_study as rs_create_study, Direction, Study};
use rustuna_core::trial::{Trial, TrialStateValues};
use rustuna_samplers::tpe::TpeSampler;
use std::ffi::{c_void, CStr};
use std::os::raw::{c_char, c_double, c_int, c_uint};
use std::slice;
use std::sync::{Arc, Mutex};

#[repr(C)]
pub enum TunaDirection {
    TunaDirectionMinimize,
    TunaDirectionMaximize,
}

#[repr(C)]
pub enum TunaSamplerKind {
    Random,
    Tpe,
}

#[repr(C)]
pub struct TunaSampler {
    sampler: *mut c_void,
    kind: TunaSamplerKind,
}

#[repr(C)]
pub struct TunaStudy {
    study: *mut c_void,
    sampler: TunaSampler,
}

#[repr(C)]
pub struct TunaTrial {
    number: c_uint,
    trial: *mut c_void,
}

#[repr(C)]
pub struct TunaPersistedTrial {
    number: c_uint,
}

#[no_mangle]
pub extern "C" fn tuna_new_tpe_sampler() -> *mut TunaSampler {
    let sampler = Box::into_raw(Box::new(TpeSampler::new())) as *mut c_void;
    let sampler = TunaSampler {
        kind: TunaSamplerKind::Tpe,
        sampler,
    };
    Box::into_raw(Box::new(sampler))
}

#[no_mangle]
pub extern "C" fn tuna_new_random_sampler() -> *mut TunaSampler {
    let sampler = Box::into_raw(Box::new(RandomSampler::new())) as *mut c_void;
    let sampler = TunaSampler {
        kind: TunaSamplerKind::Random,
        sampler,
    };
    Box::into_raw(Box::new(sampler))
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn tuna_create_study(
    study_name: *const c_char,
    sampler: TunaSampler,
    directions_ptr: *const TunaDirection,
    directions_len: c_uint,
) -> *mut TunaStudy {
    if study_name.is_null() {
        return std::ptr::null_mut();
    }
    let study_name = CStr::from_ptr(study_name).to_string_lossy().into_owned();
    let storage = InMemoryStorage::new();
    let rs_directions: Vec<Direction> =
        slice::from_raw_parts(directions_ptr, directions_len as usize)
            .iter()
            .map(|direction| match *direction {
                TunaDirection::TunaDirectionMinimize => Direction::Minimize,
                TunaDirection::TunaDirectionMaximize => Direction::Maximize,
            })
            .collect();

    let study = match rs_create_study(&study_name, storage, rs_directions) {
        Ok(study) => Box::into_raw(Box::new(study)) as *mut c_void,
        Err(_) => return std::ptr::null_mut(),
    };
    Box::into_raw(Box::new(TunaStudy { study, sampler }))
}

// TODO(c-bata): Consider introducing fn tuna_ask(study: *mut TunaStudy, trial: *mut TunaTrial) {}
// to avoid allocating the n_trials * TunaTrial objects.
#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn tuna_ask(study: *mut TunaStudy) -> *mut TunaTrial {
    if study.is_null() {
        return std::ptr::null_mut();
    }
    let study = &mut *study;
    let study_core: &mut Study = &mut *(study.study as *mut Study);

    let rs_trial = match study.sampler.kind {
        TunaSamplerKind::Random => {
            let sampler_core: Box<RandomSampler> =
                Box::from_raw(study.sampler.sampler as *mut RandomSampler);
            let sampler = Arc::new(Mutex::new(*sampler_core));
            let sampler_cloned = sampler.clone();
            let trial = study_core.ask(sampler_cloned).unwrap();
            study.sampler.sampler = Box::into_raw(Box::new(Arc::into_raw(sampler))) as *mut c_void;
            trial
        }
        TunaSamplerKind::Tpe => {
            let sampler_core: Box<TpeSampler> =
                Box::from_raw(study.sampler.sampler as *mut TpeSampler);
            let sampler = Arc::new(Mutex::new(*sampler_core));
            let sampler_cloned = sampler.clone();
            let trial = study_core.ask(sampler_cloned).unwrap();
            study.sampler.sampler = Box::into_raw(Box::new(Arc::into_raw(sampler))) as *mut c_void;
            trial
        }
    };

    let number: c_uint = rs_trial.number as c_uint;
    let trial = Box::into_raw(Box::new(rs_trial)) as *mut c_void;
    Box::into_raw(Box::new(TunaTrial { number, trial }))
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn tuna_tell(
    study: *mut TunaStudy,
    trial_number: c_int,
    values_ptr: *const c_double,
    values_len: c_uint,
) -> c_int {
    if study.is_null() {
        return -1;
    }
    let study = &mut *study;
    let study_core: &mut Study = &mut *(study.study as *mut Study);

    let values = slice::from_raw_parts(values_ptr, values_len as usize).to_vec();
    study_core
        .tell(trial_number as u32, TrialStateValues::Complete(values))
        .unwrap();
    0
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn tuna_get_n_trials(study: *mut TunaStudy) -> c_int {
    if study.is_null() {
        return -1;
    }
    let study = &mut *study;
    let study_core: &mut Study = &mut *(study.study as *mut Study);

    let n_trials = match study_core.get_trials() {
        Ok(trials) => trials.len(),
        Err(_) => return -1,
    };
    n_trials as c_int
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn tuna_get_trial(
    study: *mut TunaStudy,
    number: c_uint,
) -> *mut TunaPersistedTrial {
    if study.is_null() {
        return std::ptr::null_mut();
    }
    let study = &mut *study;
    let study_core: &mut Study = &mut *(study.study as *mut Study);
    let number = number as usize;

    // TODO(c-bata): Might be better to call storage.get_trial() instead.
    let persisted_trials = match study_core.get_trials() {
        Ok(trials) => trials,
        Err(_) => return std::ptr::null_mut(),
    };
    if number >= persisted_trials.len() {
        return std::ptr::null_mut();
    }
    let trial: &rustuna_core::trial::PersistedTrial = &persisted_trials[number];
    Box::into_raw(Box::new(TunaPersistedTrial {
        number: trial.number as c_uint,
    }))
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn tuna_suggest_float(
    trial: *mut TunaTrial,
    name: *const c_char,
    low: c_double,
    high: c_double,
    value: *mut c_double,
) -> c_int {
    if trial.is_null() || name.is_null() {
        return -1;
    }
    let trial = &mut *trial;
    let trial_core: &mut Trial = &mut *(trial.trial as *mut Trial);
    let name = CStr::from_ptr(name).to_str().unwrap();

    match trial_core.suggest_float(name, low, high) {
        Ok(v) => {
            *value = v as c_double;
            0
        }
        Err(_) => -1,
    }
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn tuna_suggest_int(
    trial: *mut TunaTrial,
    name: *const c_char,
    low: c_int,
    high: c_int,
    value: *mut c_int,
) -> c_int {
    if trial.is_null() || name.is_null() {
        return -1;
    }
    let trial = &mut *trial;
    let trial_core: &mut Trial = &mut *(trial.trial as *mut Trial);
    let name = CStr::from_ptr(name).to_str().unwrap();

    match trial_core.suggest_int(name, low as i64, high as i64) {
        Ok(v) => {
            *value = v as c_int;
            0
        }
        Err(_) => -1,
    }
}

#[no_mangle]
#[allow(clippy::missing_safety_doc)]
pub unsafe extern "C" fn tuna_suggest_categorical(
    trial: *mut TunaTrial,
    name: *const c_char,
    choices_ptr: *const *const c_char,
    choices_len: c_uint,
    value_index: *mut c_uint,
) -> c_int {
    if trial.is_null() || name.is_null() {
        return -1;
    }
    let trial = &mut *trial;
    let trial_core: &mut Trial = &mut *(trial.trial as *mut Trial);
    let name = CStr::from_ptr(name).to_str().unwrap();
    let choices: Vec<String> = slice::from_raw_parts(choices_ptr, choices_len as usize)
        .iter()
        .map(|choice| CStr::from_ptr(*choice).to_string_lossy().into_owned())
        .collect();
    match trial_core.suggest_categorical(name, &choices) {
        Ok(v) => {
            for (i, c) in choices.iter().enumerate() {
                if c == v {
                    *value_index = i as c_uint;
                    return 0;
                }
            }
            -1
        }
        Err(_) => -1,
    }
}
