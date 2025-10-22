import torch
from collections import Counter

class ExperienceBuffer():
    def __init__(self, buffer_length, batch_size, device):
        self._buffer_length = buffer_length
        self._batch_size = batch_size
        self._device = device

        self._buffer_head = 0
        self._total_samples = 0

        self._buffers = dict()
        self._flat_buffers = dict()
        self._sample_buf = torch.randperm(self._buffer_length * self._batch_size, device=self._device,
                                          dtype=torch.long)
        self._sample_buf_head = 0
        self._reset_sample_buf()

        return

    def add_buffer(self, name, buffer):
        assert(len(buffer.shape) >= 2)
        assert(buffer.shape[0] == self._buffer_length)
        assert(buffer.shape[1] == self._batch_size)
        assert(name not in self._buffers)

        self._buffers[name] = buffer
        flat_shape = [buffer.shape[0] * buffer.shape[1]] + list(buffer.shape[2:])
        self._flat_buffers[name] = buffer.view(flat_shape)

        return

    def reset(self):
        self._buffer_head = 0
        self._reset_sample_buf()
        return

    def clear(self):
        self.reset()
        self._total_samples = 0
        return

    def inc(self):
        self._buffer_head = (self._buffer_head + 1) % self._buffer_length
        self._total_samples += self._batch_size
        return

    def get_total_samples(self):
        return self._total_samples

    def get_sample_count(self):
        sample_count = min(self._total_samples, self._buffer_length * self._batch_size)
        return sample_count

    def record(self, name, data):
        assert(data.shape[0] == self._batch_size)
        data_buf = self._buffers[name]
        data_buf[self._buffer_head] = data
        return

    def get_data(self, name):
        return self._buffers[name]

    def get_data_flat(self, name):
        return self._flat_buffers[name]
    
    def set_data(self, name, data):
        data_buf = self.get_data(name)
        assert(data_buf.shape[0] == data.shape[0])
        assert(data_buf.shape[1] == data.shape[1])
        data_buf[:] = data
        return
    
    def set_data_flat(self, name, data):
        data_buf = self.get_data_flat(name)
        assert(data_buf.shape[0] == data.shape[0])
        data_buf[:] = data
        return

    def sample(self, n):
        output = dict()
        rand_idx = self._sample_rand_idx(n)

        for key, data in self._flat_buffers.items():
            batch_data = data[rand_idx]
            output[key] = batch_data

        return output

    def _reset_sample_buf(self):
        self._sample_buf[:] = torch.randperm(self._buffer_length * self._batch_size, device=self._device,
                                             dtype=torch.long)
        self._sample_buf_head = 0
        return

    def _sample_rand_idx(self, n):
        buffer_len = self._sample_buf.shape[0]
        assert(n <= buffer_len)

        if (self._sample_buf_head + n <= buffer_len):
            rand_idx = self._sample_buf[self._sample_buf_head:self._sample_buf_head + n]
            self._sample_buf_head += n
        else:
            rand_idx0 = self._sample_buf[self._sample_buf_head:]
            remainder = n - (buffer_len - self._sample_buf_head)

            self._reset_sample_buf()
            rand_idx1 = self._sample_buf[:remainder]
            rand_idx = torch.cat([rand_idx0, rand_idx1], dim=0)

            self._sample_buf_head = remainder

        sample_count = self.get_sample_count()
        rand_idx = torch.remainder(rand_idx, sample_count)
        return rand_idx
    
class ExperienceBuffer_custom(ExperienceBuffer):
    def __init__(self, buffer_length, batch_size, device):
        super().__init__(buffer_length, batch_size, device)
        self.CaT_cfg = None
        self.CaT_cmax = None
        self.CaT_prev_cmax = None
        self.CaT_tau = 0.
        self.CaT_pmax = 0.

    def init_CaT(self, CaT_cfg:dict):
            self.CaT_cfg = CaT_cfg
            self.CaT_tau = CaT_cfg['decay_tau']
            self.CaT_pmax = torch.zeros((len(CaT_cfg['constraints']),), device=self.device)

            # Basic schedule sanity check
            with torch.no_grad():
                s_l, s_h = self.CaT_cfg["schedule"]
                if s_h <= s_l:
                    raise ValueError("schedule_h must be > schedule_l")

            cons = list(self.CaT_cfg["constraints"].values())
            ids  = [c["id"] for c in cons]

            # ---- ID validation ----
            # (1) type
            if not all(isinstance(i, int) for i in ids):
                bad = [i for i in ids if not isinstance(i, int)]
                raise TypeError(f"All constraint ids must be integers; bad values: {bad}")

            # (2) duplicates
            dup_ids = [k for k, v in Counter(ids).items() if v > 1]
            if dup_ids:
                raise ValueError(f"Duplicate constraint id(s) found: {dup_ids}")

            # (3) range (needed for scatter indexing)
            n = len(cons)
            out_of_range = [i for i in ids if i < 0 or i >= n]
            if out_of_range:
                raise ValueError(f"Constraint id(s) out of range [0, {n-1}]: {out_of_range}")


            cons = list(self.CaT_cfg["constraints"].values())
            self._cat_ids = torch.tensor([c["id"] for c in cons], device=self.device)
            self._p_lo    = torch.tensor([c["p_max"][0] for c in cons], device=self.device)
            self._p_hi    = torch.tensor([c["p_max"][1] for c in cons], device=self.device)

            self.schedule_CaT()

    def process_env_step_CaT(self, CaT):
        if CaT is None:
            return
        if self.CaT_cmax is None: # initialize buffers
            self.CaT_cmax = torch.zeros(CaT.shape[-1], device=self.device)
            self.CaT_prev_cmax = torch.zeros_like(self.CaT_cmax)
        self.CaT_cmax = torch.maximum(self.CaT_cmax, CaT.amax(dim=0))

    def process_learning_step_CaT(self, it=0, CaT = None):
        '''
        Batch process CaT per learning step.
        Computes CaT_delta used to scale rewards and values.
        Returns CaT_delta
        For scheduled CaT.
        CaT starts at low p_max, reaches max p_max as training progresses
        This function, given the traininig iteration, updates p_max.
        '''
        # Constraint as Termination
        self.CaT_cmax = self.CaT_tau*self.CaT_cmax + (1-self.CaT_tau)*self.CaT_prev_cmax
        CaT_delta = (self.CaT_pmax*torch.clip(CaT/(self.CaT_cmax+1.e-6), min=0., max=1.)).max(dim=-1, keepdim=True)[0]
        assert CaT_delta.shape == self.rewards.shape, f'Shape mismatch, CaT_delta shape : {CaT_delta.shape}'
        self.CaT_prev_cmax[:] = self.CaT_cmax[:]
        if self.CaT_cfg is not None:
            s_l, s_h = self.CaT_cfg["schedule"]
            if s_h <= s_l:
                raise ValueError("schedule_h must be > schedule_l")
            # Progress alpha in [0,1]
            it_t  = torch.as_tensor(it, dtype=self.CaT_pmax.dtype, device=self.CaT_pmax.device)
            alpha = ((it_t - s_l) / float(s_h - s_l)).clamp_(0.0, 1.0)
            # Interpolate per-constraint
            vals = self._p_lo + alpha * (self._p_hi - self._p_lo)
            # Write (indexing is simpler than scatter_ for 1-D)
            self.CaT_pmax[self._cat_ids] = vals
            print(f'CaT p max : {self.CaT_pmax}')
        return CaT_delta

    def reset(self):
        super().reset()
        self.CaT_cmax.zero_()
