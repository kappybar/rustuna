use std::collections::VecDeque;

use crate::{Error, ErrorKind, Result};

pub trait TrialQueue: Send + Sync {
    fn push(&mut self, trial_id: u32) -> Result<()>;
    fn pop(&mut self) -> Result<u32>;
}

#[derive(Default)]
pub struct InMemoryTrialQueue {
    queue: VecDeque<u32>,
}

impl InMemoryTrialQueue {
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
        }
    }
}

impl TrialQueue for InMemoryTrialQueue {
    fn push(&mut self, trial_id: u32) -> Result<()> {
        self.queue.push_back(trial_id);
        Ok(())
    }

    fn pop(&mut self) -> Result<u32> {
        self.queue
            .pop_back()
            .ok_or_else(|| Error::new(ErrorKind::TrialQueueEmpty))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_and_pop() {
        let mut queue = InMemoryTrialQueue::new();

        assert!(queue.push(1).is_ok());
        assert!(queue.push(2).is_ok());
        assert!(queue.push(3).is_ok());

        assert_eq!(queue.pop().unwrap(), 3);
        assert_eq!(queue.pop().unwrap(), 2);
        assert_eq!(queue.pop().unwrap(), 1);
    }

    #[test]
    fn test_pop_empty_queue() {
        let mut queue = InMemoryTrialQueue::new();
        assert!(queue.pop().is_err());
    }
}
